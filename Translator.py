import tkinter as tk
from transformers import MarianTokenizer, MarianMTModel

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("English to Hinglish Translator")

        model_directory_path = r"C:\Users\Ameyo\OneDrive\Desktop\Hinglish-TOP-Dataset-main\Hinglish-TOP-Dataset-main\Translation model" # Replace with the actual path
        self.tokenizer = MarianTokenizer.from_pretrained(model_directory_path)
        self.model = MarianMTModel.from_pretrained(model_directory_path)

        self.input_label = tk.Label(root, text="Enter an English sentence:")
        self.input_label.pack()
        self.input_text = tk.Entry(root, width=50)
        self.input_text.pack()
        self.translate_button = tk.Button(root, text="Translate", command=self.translate_to_hinglish)
        self.translate_button.pack()
        self.output_label = tk.Label(root, text="Hinglish Translation:")
        self.output_label.pack()
        self.output_text = tk.StringVar()
        self.output_text.set("")  
        self.output_display = tk.Label(root, textvariable=self.output_text, wraplength=400, justify="left")
        self.output_display.pack()

    def translate_to_hinglish(self):
        english_sentence = self.input_text.get()

        input_ids = self.tokenizer.encode(english_sentence, return_tensors="pt")
        translation = self.model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
        hinglish_translation = self.tokenizer.decode(translation[0], skip_special_tokens=True)
        self.output_text.set(hinglish_translation)

def main():
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
