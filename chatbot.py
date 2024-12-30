import random
import json
import pickle
import numpy as np
import nltk
import sys
import io
from tensorflow.keras.models import load_model
from tkinter import Tk, Text, Scrollbar, Button, END, Entry, StringVar, Frame, Label, Canvas
# Descargar recursos de NLTK
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()
# Configurar codificación UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



# Cargar el modelo y los datos
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents2.json', encoding='utf-8').read())

# Función para procesar las entradas del usuario
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Crear una bolsa de palabras (bag of words)
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Predecir la intención
def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.3  # Reducir umbral para captar más intenciones
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else None
# Obtener una respuesta
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return "\n".join(intent['responses'])  # Combinar respuestas con saltos de línea
    return "Lo siento, no entiendo tu pregunta."

# Generar una respuesta
def generate_response(user_input):
    intent = predict_class(user_input)
    print(f"Intención detectada: {intent}")  # Depurar la intención detectada
    if intent:
        response = get_response(intent)
        print(f"Respuestas obtenidas: {response}")  # Depurar las respuestas obtenidas
        return response
    else:
        return "Lo siento, no entiendo tu pregunta. ¿Puedes intentarlo de otra manera?"



class ChatBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatBot App")
        self.root.geometry("500x600")
        self.root.configure(bg="#2c3e50")

        # Estilo general
        self.font = ("Helvetica", 12)
        self.bg_color = "#34495e"
        self.text_color = "#ecf0f1"
        self.bot_bg_color = "#34495e"
        self.user_bg_color = "#34495e"
        self.entry_bg = "#ecf0f1"
        self.entry_fg = "#2c3e50"
        self.button_color = "#1abc9c"

        # Frame principal de la conversación
        self.chat_frame = Frame(self.root, bg=self.bg_color)
        self.chat_frame.pack(pady=10, fill="both", expand=True)

        # Area de texto para la conversación
        self.text_area = Text(
            self.chat_frame,
            height=20,
            width=50,
            wrap="word",
            font=self.font,
            bg=self.bg_color,
            fg=self.text_color,
            bd=0,
            padx=10,
            pady=10,
            state="normal",  # Hacerlo completamente interactivo
            selectbackground="#1abc9c",  # Establecer el color de fondo al seleccionar
            selectforeground="white"  # Establecer el color del texto al seleccionar
        )
        self.text_area.pack(side="left", fill="both", expand=True)

        # Barra de desplazamiento
        self.scrollbar = Scrollbar(self.chat_frame, command=self.text_area.yview, bg=self.bg_color)
        self.scrollbar.pack(side="right", fill="y")
        self.text_area["yscrollcommand"] = self.scrollbar.set

        # Configuración de las etiquetas para los diferentes tipos de texto
        self.text_area.tag_config("user", foreground="white", background=self.user_bg_color, justify="right")
        self.text_area.tag_config("bot", foreground="white", background=self.bot_bg_color, justify="left")
        self.text_area.tag_config("code", foreground="black", background="#ecf0f1", font=("Courier", 10, "italic"))

        # Frame para la entrada de texto
        self.input_frame = Frame(self.root, bg=self.bg_color)
        self.input_frame.pack(pady=5, padx=10, fill="x")

        self.user_input = StringVar()
        self.entry_box = Entry(
            self.input_frame,
            textvariable=self.user_input,
            width=40,
            font=self.font,
            bg=self.entry_bg,
            fg=self.entry_fg,
            bd=2,
            relief="groove"
        )
        self.entry_box.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Botón de enviar
        self.send_button = Button(
            self.input_frame,
            text="Enviar",
            command=self.send_message,
            font=self.font,
            bg=self.button_color,
            fg=self.text_color,
            activebackground="#16a085",
            bd=0,
            padx=10,
            pady=5
        )
        self.send_button.pack(side="right")

    def send_message(self):
        user_text = self.user_input.get()
        if user_text.strip():
            self.display_message(user_text, "user")  # Mostrar mensaje del usuario
            self.root.update_idletasks()  # Asegurar actualización
            response = generate_response(user_text)
            print(f"Respuesta generada: {response}")  # Depuración
            self.display_message(response, "bot")  # Mostrar respuesta del bot
        self.user_input.set("")  # Limpiar campo de entrada

    def display_message(self, message, sender):
        print(f"Mostrando mensaje: {message} | Remitente: {sender}")  # Depurar
        tag = "user" if sender == "user" else "bot"

        if "```" in message:
            # Dividir el mensaje en texto y bloques de código
            parts = message.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    self.text_area.insert(END, f"{part.strip()}\n\n", tag)
                else:
                    # Mostrar bloque de código debajo del mensaje
                    self.text_area.insert(END, f"```\n{part.strip()}\n```\n\n", "code")
        else:
            # Mostrar solo texto normal si no contiene código
            self.text_area.insert(END, f"{message.strip()}\n\n", tag)

        # Desplazar la vista hacia abajo automáticamente
        self.text_area.yview(END)




# Ejecutar la aplicación
def main():
    root = Tk()
    app = ChatBotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
