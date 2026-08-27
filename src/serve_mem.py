import os
import sys
import argparse
import multiprocessing
from typing import Dict

from tqdm import tqdm
from flask import Flask, request, jsonify
from waitress import serve
from sentence_transformers import SentenceTransformer

from mem.retriever import Retriever


app = Flask(__name__)
server = None


class RetrievalServer:
    def __init__(self, index_dir: str, sentence_model_name: str):
        self.retrievers: Dict[str, Retriever] = {}

        sentence_model = SentenceTransformer(sentence_model_name)
        for conversation_id in tqdm(os.listdir(index_dir), desc="Loading retrievers: "):
            retriever = Retriever(sentence_model=sentence_model)
            retriever.load_index(index_dir=index_dir, conversation_id=conversation_id)
            self.retrievers[conversation_id] = retriever

    def call(self, conversation_id: str, **kwargs):
        if conversation_id not in self.retrievers:
            print(f"Conversation ID {conversation_id} not found", file=sys.stderr)
            return None

        retriever = self.retrievers[conversation_id]
        name = kwargs.get("name")

        if name == "search":
            return retriever.search(query=kwargs.get("query"), k=int(kwargs.get("k")))
        elif name == "view_session":
            return retriever.view_session(idx=int(kwargs.get("idx")))
        elif name == "view_prev_session":
            return retriever.view_prev_session(idx=int(kwargs.get("idx")))
        elif name == "view_next_session":
            return retriever.view_next_session(idx=int(kwargs.get("idx")))
        elif name == "view_sessions_by_date":
            return retriever.view_sessions_by_date(start_date=kwargs.get("start_date"), end_date=kwargs.get("end_date"))
        elif name == "find_similar_memories":
            return retriever.find_similar_memories(idx=int(kwargs.get("idx")), k=int(kwargs.get("k")))
        else:
            print(f"Invalid name {name}", file=sys.stderr)
            return None


@app.route("/")
def index():
    usage = {
        "/search": "conversation_id,query,k",
        "/view_session": "conversation_id,idx",
        "/view_prev_session": "conversation_id,idx",
        "/view_next_session": "conversation_id,idx",
        "/view_sessions_by_date": "conversation_id,start_date,end_date",
        "/find_similar_memories": "conversation_id,idx,k",
    }
    return jsonify(usage)


@app.route("/search")
def search():
    result = server.call(
        conversation_id=request.args.get("conversation_id"),
        name="search",
        query=request.args.get("query"),
        k=request.args.get("k"),
    )
    return jsonify(result)


@app.route("/view_session")
def view_session():
    result = server.call(
        conversation_id=request.args.get("conversation_id"),
        name="view_session",
        idx=request.args.get("idx"),
    )
    return jsonify(result)


@app.route("/view_prev_session")
def view_prev_session():
    result = server.call(
        conversation_id=request.args.get("conversation_id"),
        name="view_prev_session",
        idx=request.args.get("idx"),
    )
    return jsonify(result)


@app.route("/view_next_session")
def view_next_session():
    result = server.call(
        conversation_id=request.args.get("conversation_id"),
        name="view_next_session",
        idx=request.args.get("idx"),
    )
    return jsonify(result)


@app.route("/view_sessions_by_date")
def view_sessions_by_date():
    result = server.call(
        conversation_id=request.args.get("conversation_id"),
        name="view_sessions_by_date",
        start_date=request.args.get("start_date"),
        end_date=request.args.get("end_date"),
    )
    return jsonify(result)


@app.route("/find_similar_memories")
def find_similar_memories():
    result = server.call(
        conversation_id=request.args.get("conversation_id"),
        name="find_similar_memories",
        idx=request.args.get("idx"),
        k=request.args.get("k"),
    )
    return jsonify(result)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--index_dir", type=str, required=True)
    args.add_argument("--sentence_model_name", type=str, default="all-MiniLM-L6-v2")
    args.add_argument("--host", type=str, default="0.0.0.0")
    args.add_argument("--port", type=int, default=5632)
    args = args.parse_args()

    server = RetrievalServer(index_dir=args.index_dir, sentence_model_name=args.sentence_model_name)

    cores = multiprocessing.cpu_count()

    serve(
        app,
        host=args.host,
        port=args.port,
        threads=cores,
        expose_tracebacks=True,
        channel_timeout=10,
        cleanup_interval=10,
    )
