from transformers import pipeline, set_seed

def generate_poem(topic, model_name="gpt2", max_len=50, num_poems=1):
    """
    Generates a short poem on a given topic using a pre-trained model.

    Args:
        topic (str): The topic for the poem.
        model_name (str): The name of the pre-trained model to use (e.g., "gpt2", "gpt2-medium").
        max_len (int): The maximum length of the generated poem.
        num_poems (int): Number of poems to generate.

    Returns:
        list: A list of generated poem strings.
    """
    try:
        # Initialize the text generation pipeline
        # Using a specific task makes it easier if the model supports it.
        # For general text generation, 'text-generation' is good.
        generator = pipeline('text-generation', model=model_name)
        print(f"Model '{model_name}' loaded. Generating poem...")

        # Set a seed for reproducibility (optional, but good for consistent results during testing)
        set_seed(42)

        # Craft a prompt. This is crucial for good results!
        # We can be more specific to guide the model.
        prompt = f"Compose a short, creative poem about {topic}:\n\n"

        generated_texts = generator(
            prompt,
            max_length=max_len + len(prompt.split()), # Adjust max_length to account for prompt length
            min_length=10 + len(prompt.split()),     # Ensure it's not too short
            num_return_sequences=num_poems,
            num_beams=5,  # Beam search can lead to more coherent text
            no_repeat_ngram_size=2, # Prevents repeating the same sequence of 2 words
            early_stopping=True,
            temperature=0.7 # Lower temperature for less random, more focused output
        )

        poems = []
        for i, output in enumerate(generated_texts):
            # The output includes the prompt, so we need to remove it.
            poem_text = output['generated_text'][len(prompt):].strip()
            # Sometimes models add extra text or self-correction; try to clean it up.
            poem_text = poem_text.split("\n\n")[0] # Take only the first stanza if multiple are generated
            poems.append(f"--- Poem {i+1} ---\n{poem_text}")

        return poems

    except Exception as e:
        return [f"Error generating poem: {e}"]

if __name__ == "__main__":
    print("Welcome to the Generative AI Poem Creator!")
    user_topic = input("Enter a topic for your poem (e.g., 'the ocean', 'a lonely star'): ")

    if user_topic:
        # You can experiment with different models. "gpt2" is smaller and faster.
        # "gpt2-medium" or "gpt2-large" might give better results but require more resources.
        generated_poems = generate_poem(user_topic, model_name="gpt2", max_len=60, num_poems=1)
        for poem in generated_poems:
            print("\n" + poem)
    else:
        print("No topic provided. Exiting.")