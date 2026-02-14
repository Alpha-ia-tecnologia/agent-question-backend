import base64
import os
import uuid

def save_image(image_base64: str):
    path = os.path.abspath("export/images")
    os.makedirs(path, exist_ok=True)  # Garante que a pasta existe
    file_name = str(uuid.uuid4().hex)
    file_path = f"{path}/{file_name}.png"
    image_data = base64.b64decode(image_base64)
    with open(file_path, "wb") as img_file:
        img_file.write(image_data)
        img_file.close()
    return file_path