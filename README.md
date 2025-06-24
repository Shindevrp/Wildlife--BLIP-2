# Wildlife BLIP-2 Fine-Tuning Project

This project demonstrates how to fine-tune the BLIP-2 model for wildlife image captioning and visual question answering (VQA) using your own dataset, and provides a user-friendly web interface for testing your model.

## Folder Structure

```
project/
│
├── images/             # Place your images here
│   ├── elephant.jpg
│   ├── deer.jpg
│   └── ...
│
├── data.json           # Annotations file (see format below)
├── finetune_blip2.py   # Fine-tuning script
├── app.py              # Streamlit web app for inference
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Dataset Format

Example `data.json`:

```
[
  {
    "image": "images/elephant.jpg",
    "caption": "A herd of elephants crossing the river."
  },
  {
    "image": "images/deer.jpg",
    "question": "Is there a deer in the image?",
    "answer": "Yes"
  }
]
```

- For image captioning, use the `caption` field.
- For VQA, use `question` and `answer` fields.

## Setup Environment

Install the required packages:

```bash
pip install -r requirements.txt
```

## Fine-Tuning

Run the fine-tuning script:

```bash
python finetune_blip2.py
```

## Web Frontend (Streamlit)

Run the Streamlit app for batch image upload, gallery, and history features:

```bash
streamlit run app.py
```

- Upload one or more images.
- Select Caption or VQA mode.
- For VQA, enter your question.
- View results, gallery, and history in the web UI.

## Inference Example (Python)

After training, you can generate captions or answers using the model in Python. See `finetune_blip2.py` for details.

## Tips
- Add more images and annotations to improve performance.
- You can use the same script for both captioning and VQA tasks.
- For deployment, consider converting the model to TorchScript or ONNX for edge devices.

---
