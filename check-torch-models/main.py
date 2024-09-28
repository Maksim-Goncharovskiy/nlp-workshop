import json
from utils import get_accuracy


def check_torch_model(model_name: str, model_hugging_face_name: str):
    accuracy = get_accuracy(model_name, model_hugging_face_name)
    return accuracy


if __name__ == "__main__":
    with open('test.json', 'r') as js_file:

        js_data = json.load(js_file)

        model_name = js_data['model_name']
        model_hugging_face_name = js_data['model_hugging_face_name']

        result = check_torch_model(model_name, model_hugging_face_name)

        print(f"{model_name} accuracy score: {result}")

        js_file.close()
