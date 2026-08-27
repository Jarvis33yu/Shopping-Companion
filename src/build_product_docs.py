import sys
import argparse
import ujson as json

import common_io
from tqdm import tqdm

from util.misc import is_rubbish_kv


class BuildProductDocs:
    def __init__(self, args):
        self.args = args
        self.selected_cols = [
            "product_id",
            "product_name",
            "seller_id",
            "category",
            "price",
            "attributes",
            "options",
        ]

    def download(self):
        project = self.args.odps_table.split(".", 1)[0]
        table = self.args.odps_table.split(".", 1)[1]

        reader = common_io.table.TableReader(
            f"odps://{project}/tables/{table}",
            selected_cols=",".join(self.selected_cols),
            slice_id=0,
            slice_count=1,
        )

        total_records_num = reader.get_row_count()
        batch_size = 100
        last_batch_size = total_records_num % batch_size
        print(f"total_records_num: {total_records_num}, batch_size: {batch_size}, last_batch_size: {last_batch_size}", file=sys.stderr)

        pbar = tqdm(total=total_records_num, desc="Downloading product data: ")
        download = 0
        with open(self.args.products_file, "w") as fout:
            while True:
                try:
                    batch = reader.read(batch_size)
                    for row in batch:
                        pbar.update(1)
                        data = {key: row[i] for i, key in enumerate(self.selected_cols)}

                        fout.write(json.dumps(data) + "\n")
                        download += 1
                    if download == total_records_num - last_batch_size and last_batch_size > 0:
                        batch_size = last_batch_size
                except common_io.exception.OutOfRangeException:
                    reader.close()
                    break

        print(f"Downloaded {download} products", file=sys.stderr)

    def convert_products_to_docs(self):
        with open(self.args.products_file, "r") as fin, open(self.args.docs_file, "w") as fout:
            for line in tqdm(fin, desc="Converting products to docs: "):
                data = json.loads(line.strip())

                product_id = data["product_id"]
                product_name = data["product_name"]
                seller_id = data["seller_id"]
                category = data["category"]
                price = data["price"]
                attributes = data["attributes"]
                options = data["options"]

                # parse attributes
                attr_dict = {}
                for kv in attributes.split(chr(3)):
                    splited = kv.split(chr(2))
                    if len(splited) != 2:
                        continue
                    k, vs = splited
                    vs = vs.split(chr(1))
                    if any(is_rubbish_kv(k, v) for v in vs):
                        continue
                    attr_dict[k] = vs

                # parse options
                option_dicts = []
                for kvs in options.split(chr(4)):
                    option_dict = {}
                    for kv in kvs.split(chr(3)):
                        splited = kv.split(chr(2))
                        if len(splited) != 2:
                            continue
                        k, vs = splited
                        vs = vs.split(chr(1))
                        if any(is_rubbish_kv(k, v) for v in vs):
                            continue
                        option_dict[k] = vs
                    option_dicts.append(option_dict)

                # content
                content = []

                content.append(product_name)

                kv_set = set()
                if attr_dict:
                    for k, v in attr_dict.items():
                        v_str = ", ".join(v)
                        if (k, v_str) in kv_set:
                            continue
                        kv_set.add((k, v_str))
                        content.append(f"{k}: {v_str}")

                if option_dicts:
                    for option_dict in option_dicts:
                        for k, v in option_dict.items():
                            v_str = ", ".join(v)
                            if (k, v_str) in kv_set:
                                continue
                            kv_set.add((k, v_str))
                            content.append(f"{k}: {v_str}")

                content_str = "\n".join(content)

                # product
                product = {k: data[k] for k in self.selected_cols if data[k]}

                if attr_dict:
                    product["attributes"] = attr_dict
                elif "attributes" in product:
                    del product["attributes"]

                if option_dicts:
                    product["options"] = option_dicts
                elif "options" in product:
                    del product["options"]

                # doc
                doc = {
                    "id": product_id,
                    "contents": content_str,
                    "product": product,
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--tasks", type=str, required=True)
    args.add_argument("--odps_table", type=str)
    args.add_argument("--products_file", type=str)
    args.add_argument("--docs_file", type=str)
    args = args.parse_args()

    builder = BuildProductDocs(args)
    if "download" in args.tasks.split(","):
        assert args.odps_table is not None
        assert args.products_file is not None
        builder.download()
    if "convert" in args.tasks.split(","):
        assert args.products_file is not None
        assert args.docs_file is not None
        builder.convert_products_to_docs()
