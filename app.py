from openai import OpenAI
import streamlit as st
from PIL import Image
from datetime import datetime
import tensorflow as tf
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Crop Doctor",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# OPENROUTER AI SETUP
# -----------------------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=""
)

# -----------------------------
# LOAD AI MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "model/crop_disease_model.h5",
    compile=False
)

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold'
]

# -----------------------------
# CUSTOM UI
# -----------------------------

st.markdown("""
<style>

/* HIDE STREAMLIT ELEMENTS */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(-45deg,
        #000000,
        #050816,
        #0f172a,
        #071b2b,
        #001f1f);
    background-size: 500% 500%;
    animation: bgmove 18s ease infinite;
    color: white;
}

@keyframes bgmove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* MAIN CONTAINER */
.block-container {
    background: rgba(255,255,255,0.04);
    border-radius: 25px;
    padding: 2rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 0 15px rgba(0,255,170,0.08);
}

/* MAIN TITLE */
.title {
    text-align: center;
    font-size: 70px !important;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 10px;
    color: #00ffbf;

    line-height: 1;

    text-shadow:
        0 0 10px #00ffbf,
        0 0 22px #00ffbf,
        0 0 45px #00ffbf,
        0 0 70px #00ffbf;

    animation: floating 3s ease-in-out infinite;
}

@keyframes floating {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
    100% {transform: translateY(0px);}
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.80);
    font-size: 22px;
    margin-bottom: 35px;
}

/* TEXT */
p, label, div, span {
    color: #e5e7eb !important;
    font-weight: 500;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.75);
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] {
    background: black !important;
    border: 2px solid #00ffff !important;
    border-radius: 16px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    color: white !important;
    font-weight: bold !important;
}

.stSelectbox input {
    color: white !important;
}

.stSelectbox span {
    color: white !important;
    font-weight: bold !important;
}

.stSelectbox svg {
    fill: white !important;
}

/* DROPDOWN OPTIONS */
div[data-baseweb="select"] > div {
    background-color: #001f2f !important;
    color: white !important;
}

ul {
    background-color: #001f2f !important;
}

li {
    color: white !important;
    font-weight: bold !important;
}

li:hover {
    background-color: #00aaff !important;
    color: black !important;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    background: rgba(0,20,20,0.90) !important;
    border: 2px solid #00ffbf !important;
    border-radius: 18px;
    padding: 16px;
}

/* UPLOAD AREA */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,0,0,0.55) !important;
    border: 2px dashed #00ffbf !important;
    border-radius: 16px !important;
}

/* BROWSE BUTTON */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(to right,#00ffbf,#00aaff) !important;
    color: black !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: bold !important;
}

/* INPUT BOX */
.stTextInput>div>div>input {
    background: rgba(0,20,20,0.90) !important;
    color: #b6fff0 !important;
    border: 2px solid #00ffbf !important;
    border-radius: 14px;
    padding: 12px;
    font-size: 16px;
    font-weight: bold;
}

/* BUTTONS */
.stButton>button {
    width: 100%;
    background: linear-gradient(to right,#00ffbf,#00aaff);
    color: black;
    border: none;
    border-radius: 14px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
}

/* CHAT BOX */
.chat-box {
    background: rgba(0,255,191,0.08);
    border: 2px solid #00ffbf;
    padding: 18px;
    border-radius: 18px;
    margin-top: 15px;
    color: #dffff7 !important;
    font-size: 17px;
    font-weight: 500;
    box-shadow: 0 0 12px rgba(0,255,191,0.2);
    
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* REALISTIC 3D IMAGE */

img {

    border-radius: 25px !important;

    transition:
    transform 0.15s ease,
    box-shadow 0.2s ease !important;

    transform-style: preserve-3d !important;

    will-change: transform;

    box-shadow:
    0 25px 45px rgba(0,0,0,0.45),
    0 0 30px rgba(0,255,191,0.25);
}

/* IMAGE CONTAINER */

.element-container:has(img){

    perspective: 1200px;
}

/* BUTTON 3D */

.stButton>button{

    transition:
    transform 0.15s ease,
    box-shadow 0.2s ease !important;

    transform-style: preserve-3d;

    box-shadow:
    0 10px 25px rgba(0,255,191,0.35);
}

.stButton>button:hover{

    transform:
    rotateX(15deg)
    rotateY(-15deg)
    translateY(-5px)
    scale(1.04);

    box-shadow:
    0 20px 40px rgba(0,255,191,0.55);
}

/* CHAT BOX */

.chat-box{

    transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;

    transform-style: preserve-3d;
}

.chat-box:hover{

    transform:
    rotateX(6deg)
    rotateY(-6deg)
    scale(1.02);

    box-shadow:
    0 20px 45px rgba(0,255,191,0.35);
}

</style>

<script>

document.addEventListener("mousemove", function(e){

    const imgs = document.querySelectorAll("img");

    imgs.forEach((img)=>{

        const rect = img.getBoundingClientRect();

        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateY = ((x - centerX) / centerX) * 25;
        const rotateX = ((centerY - y) / centerY) * 25;

        img.style.transform =
        `
        rotateX(${rotateX}deg)
        rotateY(${rotateY}deg)
        scale(1.05)
        `;
    });

});

document.addEventListener("mouseleave", function(){

    const imgs = document.querySelectorAll("img");

    imgs.forEach((img)=>{

        img.style.transform =
        "rotateX(0deg) rotateY(0deg) scale(1)";
    });

});

</script>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------

st.markdown(
    '<p class="title">🌱 AI Crop Doctor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Smart Agriculture Assistant with AI</p>',
    unsafe_allow_html=True
)

# -----------------------------
# VOICE FUNCTION
# -----------------------------

def speak(text):

    clean_text = text.replace("\n", " ").replace('"', '')

    js_code = f"""
    <script>

    window.speechSynthesis.cancel();

    var msg = new SpeechSynthesisUtterance("{clean_text}");

    msg.lang = "{'te-IN' if language == 'Telugu' else 'en-US'}";

    msg.volume = 1;
    msg.rate = 0.9;
    msg.pitch = 1;

    speechSynthesis.speak(msg);

    </script>
    """

    st.components.v1.html(js_code, height=0)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🌿 Navigation")

language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Telugu"]
)

st.sidebar.success("AI Agriculture Assistant")

# -----------------------------
# WEATHER
# -----------------------------

st.subheader("🌦 Weather Suggestion")

weather = "Sunny"

st.success(f"Current Weather: {weather}")

st.info("☀️ Perfect weather for farming and crop drying.")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

st.markdown("---")

st.header("📷 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose crop image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        width=350,
        caption="Uploaded Crop Leaf"
    )

    analyze = st.button("🔍 Analyze Disease")

    if analyze:

        # -----------------------------
        # AI PREDICTION
        # -----------------------------

        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing Crop Disease..."):

            prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = int(np.max(prediction) * 100)

        disease_name = class_names[predicted_class]

        # -----------------------------
        # AI DISEASE EXPLANATION
        # -----------------------------

        with st.spinner("Generating AI Disease Report..."):

            disease_prompt = f"""
            You are an expert agriculture doctor.

            If Selected Language is Telugu,
            answer only in Telugu script.

            If Selected Language is English,
            answer only in English.

            Selected Language:
            {language}

            Explain this crop disease clearly and professionally.

            Disease Name:
            {disease_name}

            Give:
            1. Disease severity
            2. Causes
            3. Symptoms
            4. Treatment
            5. Prevention
            6. Farmer advice

            Use simple language.
            """

            response = client.chat.completions.create(
                model="microsoft/phi-4-mini-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": disease_prompt
                    }
                ]
            )

            disease_answer = response.choices[0].message.content

            st.session_state.disease_result = disease_answer
            st.session_state.disease_name = disease_name
            st.session_state.confidence = confidence

    if "disease_result" in st.session_state:

        st.markdown("---")

        st.success(
            f"🌱 Disease Detected: {st.session_state.disease_name}"
        )

        st.progress(st.session_state.confidence)

        st.write(
            f"### Confidence Level: {st.session_state.confidence}%"
        )

        st.markdown(
            f'''
            <div class="chat-box">
            {st.session_state.disease_result}
            </div>
            ''',
            unsafe_allow_html=True
        )

        if "disease_voice" not in st.session_state:
            st.session_state.disease_voice = ""

        st.session_state.disease_voice = st.session_state.disease_result

        if st.button(
            "🔊 Speak Result",
            key="speak_result"
        ):

            speak(st.session_state.disease_voice)

# -----------------------------
# AI CHATBOT
# -----------------------------

st.markdown("---")

st.header("💬 AI Farming Assistant")

user_question = st.text_input(
    "Ask your farming question..."
)

if user_question:

    with st.spinner("🤖 AI Thinking..."):

        prompt = f"""
        You are an expert agriculture assistant.

        If Selected Language is Telugu,
        answer only in Telugu script.

        If Selected Language is English,
        answer only in English.

        Selected Language:
        {language}

        Give smart farming answers.

        Explain clearly and simply.

        User Question:
        {user_question}
        """

        response = client.chat.completions.create(
            model="microsoft/phi-4-mini-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

    st.markdown(
        f'''
        <div class="chat-box">
        {answer}
        </div>
        ''',
        unsafe_allow_html=True
    )

    if st.button(
        "🔊 Speak Chatbot Answer",
        key="ai_voice"
    ):

        speak(answer)

# -----------------------------
# STOP VOICE BUTTON
# -----------------------------

if st.button(
    "🛑 Stop Voice",
    key="stop_voice"
):

    stop_js = """
    <script>

    window.speechSynthesis.cancel();

    </script>
    """

    st.components.v1.html(
        stop_js,
        height=0
    )

# -----------------------------
# FARMING TIPS
# -----------------------------

st.markdown("---")

st.subheader("🌾 Top 5 Farming Tips")

farm_tips = [
    "💧 Use drip irrigation to save water.",
    "🌿 Check leaves regularly for diseases.",
    "☀️ Give plants enough sunlight daily.",
    "🌱 Use organic compost for healthy growth.",
    "🚜 Remove weeds frequently."
]

for tip in farm_tips:
    st.success(tip)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

current_time = datetime.now().strftime("%d-%m-%Y %H:%M")

st.caption(f"🕒 {current_time}")

st.caption("Built with using Python + Streamlit")