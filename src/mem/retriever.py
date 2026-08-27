import os
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from util.misc import convert_date_to_timestamp


class ConversationDTO(BaseModel):
    question_id: str
    question_type: str
    question: str
    answer: Any
    question_date: str
    haystack_dates: List[str]
    haystack_session_ids: List[str]
    haystack_sessions: List[List[Dict]]
    answer_session_ids: List[str]

    lastest_purchase_date: str = ""
    repeat_purchase_cycle: str = ""
    complement_date: str = ""


class NoteDTO(BaseModel):
    conversation_id: str
    session_id: str
    number_of_sessions: int
    number_of_turns: int
    index: int
    turn_index: int
    role: str
    content: str
    date: str
    timestamp: int


class Retriever(BaseModel):
    conversation_id: str = ""
    notes: List[NoteDTO] = []
    sentence_model: Any = None
    idx2sess: Dict[int, int] = {}
    sess2indices: Dict[int, List[int]] = {}
    docs: List[str] = []
    embeddings: Any = None

    def check_arguments(self):
        assert self.conversation_id, "Conversation ID is not set"
        assert self.notes, "Notes are not set"
        assert self.sentence_model, "Sentence model is not set"
        assert self.idx2sess, "Index to session mapping is not set"
        assert self.sess2indices, "Session to indices mapping is not set"
        assert self.docs, "Documents are not set"
        assert self.embeddings is not None, "Embeddings are not set"
        assert len(self.notes) == len(self.docs) == len(self.embeddings), "Number of notes, documents, and embeddings do not match"

    def build_index(self, conversation_id: str, conversation: ConversationDTO):
        self.conversation_id = conversation_id

        # notes
        sessions = conversation.haystack_sessions
        session_ids = conversation.haystack_session_ids
        dates = conversation.haystack_dates

        assert len(sessions) == len(session_ids) == len(dates), f"Number of sessions ({len(sessions)}), session IDs ({len(session_ids)}), and dates ({len(dates)}) do not match"

        index = 0
        for session, session_id, date in zip(sessions, session_ids, dates):
            for turn_index, turn in enumerate(session):
                role = turn["role"]
                content = turn["content"]
                if role in {"customer", "user"}:
                    role = "user"
                elif role in {"seller", "assistant"}:
                    role = "assistant"
                else:
                    raise ValueError(f"Invalid role: {role}. Must be one of {{'customer', 'user', 'seller', 'assistant'}}")

                note = NoteDTO(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    number_of_sessions=len(sessions),
                    number_of_turns=len(session),
                    index=index,
                    turn_index=turn_index,
                    role=role,
                    content=content,
                    date=date,
                    timestamp=convert_date_to_timestamp(date),
                )
                self.notes.append(note)
                index += 1

        # embeddings
        sess_id2idx = {}
        for note in self.notes:
            if note.session_id not in sess_id2idx:
                sess_idx = len(sess_id2idx)
                sess_id2idx[note.session_id] = sess_idx

            sess_idx = sess_id2idx[note.session_id]
            doc_idx = len(self.docs)

            self.idx2sess[doc_idx] = sess_idx

            if sess_idx not in self.sess2indices:
                self.sess2indices[sess_idx] = []
            self.sess2indices[sess_idx].append(doc_idx)

            self.docs.append(note.content)

        self.embeddings = self.sentence_model.encode(self.docs)

        self.check_arguments()

    def save_index(self, index_dir: str):
        self.check_arguments()

        base_path = os.path.join(index_dir, self.conversation_id)
        os.makedirs(base_path, exist_ok=True)

        np.save(os.path.join(base_path, "embeddings.npy"), self.embeddings)

        pkl = {
            "conversation_id": self.conversation_id,
            "notes": [note.model_dump(mode="json") for note in self.notes],
            "idx2sess": self.idx2sess,
            "sess2indices": self.sess2indices,
            "docs": self.docs,
        }
        with open(os.path.join(base_path, "index.pkl"), "wb") as fout:
            pickle.dump(pkl, fout)

    def load_index(self, index_dir: str, conversation_id: str):
        base_path = os.path.join(index_dir, conversation_id)

        self.embeddings = np.load(os.path.join(base_path, "embeddings.npy"))

        with open(os.path.join(base_path, "index.pkl"), "rb") as fin:
            pkl = pickle.load(fin)
            self.conversation_id = pkl["conversation_id"]
            self.notes = [NoteDTO(**note) for note in pkl["notes"]]
            self.idx2sess = pkl["idx2sess"]
            self.sess2indices = pkl["sess2indices"]
            self.docs = pkl["docs"]

        self.check_arguments()

    def format_search_results(self, indices: List[int]) -> str:
        results = []
        for idx in indices:
            note = self.notes[idx]
            results.append(f"[Date: {note.date}, Index: {idx}] {note.role}: {note.content}")
        delimiter = "\n\n" + "-" * 10 + "\n\n"
        return delimiter.join(results).strip()

    def format_session_results(self, indices: List[int]) -> str:
        results = []
        cur_date = ""
        for idx in indices:
            note = self.notes[idx]
            if not cur_date:
                cur_date = note.date
                results.append(f"[Date: {cur_date}]")
            elif cur_date != note.date:
                cur_date = note.date
                results.append(f"\n[Date: {cur_date}]")
            results.append(f"[Index: {idx}] {note.role}: {note.content}")
        return "\n".join(results).strip()

    def search(self, query: str, k: int) -> Tuple[List[int], str]:
        self.check_arguments()

        query_embedding = self.sentence_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return top_k_indices.tolist(), self.format_search_results(top_k_indices)

    def view_session(self, idx: int) -> Tuple[List[int], str]:
        self.check_arguments()

        if idx not in self.idx2sess:
            return [], f"Memory index '{idx}' not exists"

        indices = self.sess2indices[self.idx2sess[idx]]
        return indices, self.format_session_results(indices)

    def view_prev_session(self, idx: int) -> Tuple[List[int], str]:
        self.check_arguments()

        if idx not in self.idx2sess:
            return [], f"Memory index '{idx}' not exists"

        prev_sess_idx = self.idx2sess[idx] - 1

        if prev_sess_idx < 0:
            indices = []
        else:
            indices = self.sess2indices[prev_sess_idx]
        return indices, self.format_session_results(indices)

    def view_next_session(self, idx: int) -> Tuple[List[int], str]:
        self.check_arguments()

        if idx not in self.idx2sess:
            return [], f"Memory index '{idx}' not exists"

        next_sess_idx = self.idx2sess[idx] + 1

        if next_sess_idx >= len(self.sess2indices):
            indices = []
        else:
            indices = self.sess2indices[next_sess_idx]

        return indices, self.format_session_results(indices)

    def view_sessions_by_date(self, start_date: str, end_date: str) -> Tuple[List[List[int]], List[str]]:
        self.check_arguments()

        start_timestamp = convert_date_to_timestamp(start_date)
        end_timestamp = convert_date_to_timestamp(end_date)

        indices = []
        for idx, note in enumerate(self.notes):
            if note.timestamp >= start_timestamp and note.timestamp <= end_timestamp:
                indices.append(idx)

        indices.sort()

        sess_indices = []
        cur_sess_idx = None
        cur_indices = []
        for idx in indices:
            sess_idx = self.idx2sess[idx]
            if cur_sess_idx is None:
                cur_sess_idx = sess_idx
                cur_indices.append(idx)
            elif cur_sess_idx == sess_idx:
                cur_indices.append(idx)
            else:
                sess_indices.append(cur_indices)
                cur_sess_idx = sess_idx
                cur_indices = [idx]

        if cur_indices:
            sess_indices.append(cur_indices)
        return sess_indices, [self.format_session_results(indices) for indices in sess_indices]

    def find_similar_memories(self, idx: int, k: int) -> Tuple[List[int], str]:
        self.check_arguments()

        query_embedding = self.embeddings[idx]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-(k+1):][::-1]
        top_k_indices = top_k_indices[1:]

        return top_k_indices.tolist(), self.format_search_results(top_k_indices)
