import streamlit as st
from transformers import pipeline, set_seed
import torch # Good to import explicitly if checking for GPU

# --- Helper Function to Load Model (for Caching) ---
# @st.cache_resource is used to cache resources like models so they don't reload on every interaction.
@st.cache_resource
def load_generator_pipeline(model_name="gpt2"):
    """Loads and caches the Hugging Face pipeline."""
    print(f"Loading model: {model_name}...") # This will print to your console, not the web UI
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU
    try:
        generator = pipeline('text-generation', model=model_name, device=device)
        print(f"Model '{model_name}' loaded successfully on {'GPU' if device == 0 else 'CPU'}.")
        return generator
    except Exception as e:
        # If model loading fails, display an error in Streamlit and stop.
        st.error(f"Fatal Error: Could not load model '{model_name}'. Exception: {e}")
        st.stop() # Stop the app execution if model can't be loaded.
        return None


# --- Your Original Poem Generation Logic (slightly adapted) ---
def generate_poem_for_streamlit(generator_pipeline, topic, max_len=60, num_poems=1, temperature=0.7, num_beams=5, seed_value=42):
    """
    Generates poem(s) using the provided pipeline and parameters.
    Returns a list of poem strings.
    """
    if generator_pipeline is None:
        return ["Error: Model pipeline not available."]

    try:
        set_seed(seed_value) # Set seed for reproducibility

        prompt = f"Compose a short, creative poem about {topic}:\n\n"

        generated_texts = generator_pipeline(
            prompt,
            max_length=max_len + len(prompt.split()),
            min_length=10 + len(prompt.split()),
            num_return_sequences=num_poems,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=temperature
        )

        poems = []
        for i, output in enumerate(generated_texts):
            poem_text = output['generated_text'][len(prompt):].strip()
            poem_text = poem_text.split("\n\n")[0]
            # You can choose to keep or remove the "--- Poem X ---" prefix
            # For a cleaner UI, you might just return the poem_text directly
            # poems.append(f"--- Poem {i+1} ---\n{poem_text}")
            poems.append(poem_text) # Returning just the poem text

        return poems

    except Exception as e:
        return [f"Error generating poem: {e}"]

# --- Streamlit User Interface ---
st.set_page_config(page_title="AI Poem Generator", layout="wide")
st.title("✒️ Generative AI Poem Creator")
st.markdown("Enter a topic below and let the AI craft a poem for you!")

# --- Sidebar for Controls (Optional but good for organization) ---
with st.sidebar:
    st.header("⚙️ Controls")
    # For this direct conversion, we'll keep model fixed, but you could add a selectbox
    # model_choice = st.selectbox("Choose Model", ("gpt2", "gpt2-medium")) # Example
    model_name_to_load = "gpt2" # Keeping it simple as per original script

    # We can make other parameters interactive later if desired
    max_len_param = st.slider("Max Poem Length (approx words):", 30, 150, 60, 10)
    temp_param = st.slider("Creativity (Temperature):", 0.5, 1.0, 0.7, 0.05)
    num_poems_param = st.number_input("Number of Poems to Generate:", 1, 5, 1)
    seed_param = st.number_input("Seed (for reproducibility):", 0, 1000, 42)


# --- Main Page ---
user_topic = st.text_input("Enter a topic for your poem (e.g., 'the ocean', 'a lonely star'):", "a silent river")

# Button to trigger poem generation
if st.button("✨ Generate Poem ✨", type="primary"):
    if user_topic:
        # Load the model (it will be cached after the first run for this model_name)
        with st.spinner(f"Loading AI model ({model_name_to_load})... This might take a moment on first run."):
            active_pipeline = load_generator_pipeline(model_name=model_name_to_load)

        if active_pipeline:
            with st.spinner(f"AI is composing your poem(s) about '{user_topic}'..."):
                generated_poems = generate_poem_for_streamlit(
                    active_pipeline,
                    user_topic,
                    max_len=max_len_param,
                    num_poems=num_poems_param,
                    temperature=temp_param,
                    num_beams=5, # Kept fixed from original, could be a slider
                    seed_value=seed_param
                )

            st.subheader(f"📜 Your Poem(s) about '{user_topic.capitalize()}'")
            if "Error" in generated_poems[0]:
                st.error(generated_poems[0])
            else:
                for i, poem in enumerate(generated_poems):
                    if num_poems_param > 1:
                        st.markdown(f"--- **Poem {i+1}** ---")
                    # Using st.text_area makes it easy to copy and preserves formatting
                    st.text_area(label=f"Poem Output {i+1}" if num_poems_param > 1 else "Poem Output", value=poem, height=150 + poem.count('\n')*20, disabled=True, label_visibility="collapsed")
                    st.markdown("---") # Visual separator
    else:
        st.warning("Please enter a topic for your poem.")

st.markdown("---")
st.caption("Powered by Hugging Face Transformers & Streamlit")