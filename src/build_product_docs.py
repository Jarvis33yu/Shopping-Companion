import argparse
import ujson as json

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
    args.add_argument("--products_file", type=str, required=True)
    args.add_argument("--docs_file", type=str, required=True)
    args = args.parse_args()

    builder = BuildProductDocs(args)
    builder.convert_products_to_docs()
