import ujson as json


class BaseTool:
    name: str
    description: str
    parameters: dict[str, str]

    def execute(self, **kwargs):
        raise NotImplementedError()

    def to_qwen3_string(self):
        d = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        return json.dumps(d)

    def to_string(self):
        return f"Name: {self.name}\nDescription: {self.description}\nParameters: {json.dumps(self.parameters)}"
