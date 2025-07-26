# Satellite Image Change Detection with Deep Learning
Used dataset: https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd?resource=download
## For More Training and Evaluating Processes : Go to Documentation_ENG.pdf or Documentation_TR.pdf
This project is a deep learning-based solution designed to detect changes between two satellite images taken at different times. It focuses on identifying changes such as new buildings, demolitions, and natural landscape transformations using the [LEVIR-CD](https://github.com/levircd/LEVIR-CD) dataset.

The model can be used in **three different ways**:
- Via **command-line (bash)**,
- Through a **Streamlit web interface**,
- Or using a **FastAPI service**.

---

## 🧠 Model Objective

The main goal of this model is to detect and visualize differences between satellite images captured at two different timestamps. This includes:
- **Urban changes** (e.g., building construction or demolition),
- **Environmental shifts** (e.g., vegetation growth or removal),
- And other structural transformations.

---

## 🗂️ Setup Instructions

### 🔁 Clone the Repository

```bash
git clone https://github.com/hitaskomur/satellite_image_change_detection_with_dl.git
cd satellite_image_change_detection_with_dl
```

### 🛠️ Create & Activate Virtual Environment

```bash
python -m venv env
env\Scripts\activate.bat   # For Windows
# source env/bin/activate    # For Linux or MacOS
```

### 📦 Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Command-Line (Bash)

Run the model using:

```bash
python bashcode.py <first_image.png> <second_image.png> <output_mask.png>
```

This will process the two satellite images and generate a change mask.

---

### 2️⃣ Streamlit Web Interface

Launch the Streamlit app with:

```bash
streamlit run app.py
```

Then open the automatically generated local URL in your browser to use the interactive UI.

---

### 3️⃣ FastAPI Service

Start the FastAPI backend server with:

```bash
uvicorn fast:app --reload
```

Once running, visit:

```
http://127.0.0.1:8000/docs
```

to interact with the API through the built-in Swagger UI.

---


## 📌 Notes

- Ensure that your input images follow the same preprocessing format used during training (e.g., size, channels).
- All outputs will be saved to the path you define in the command.

---

## 📬 Here is some pictures
<img width="1082" height="741" alt="streamlit_1" src="https://github.com/user-attachments/assets/26b25421-e79d-48bc-aff0-6eabd4d28cad" />

<img width="1131" height="811" alt="streamlit_2" src="https://github.com/user-attachments/assets/c2814e0b-0cb1-4d21-ad27-30344b1e71e2" />

<img width="1835" height="972" alt="fastapi" src="https://github.com/user-attachments/assets/d0e895cf-d6e8-46a8-bad1-db3b7bf3f950" />


