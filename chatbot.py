import random
import json
import pickle
import numpy as np
import nltk
import sys
import io
from tensorflow.keras.models import load_model
from tkinter import Tk, Text, Scrollbar, Button, END, Entry, StringVar, Frame, Label, Canvas, LEFT, RIGHT
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
        self.root.configure(bg="#1a1a2e")  # Fondo principal oscuro

        # Estilo general
        self.font = ("Helvetica", 12)
        self.bg_color = "#16213e"  # Azul profundo
        self.text_color = "#ffffff"  # Texto blanco
        self.bot_bg_color = "#4c00ff"  # Púrpura vibrante para el bot
        self.user_bg_color = "#00a8cc"  # Azul claro para el usuario
        self.entry_bg = "#e8e8e8"  # Gris claro para entrada
        self.entry_fg = "#1a1a2e"  # Texto oscuro en entrada
        self.button_color = "#ffdc00"  # Amarillo brillante para botón

        # Frame principal para el canvas y scrollbar
        self.chat_frame_container = Frame(self.root, bg=self.bg_color)
        self.chat_frame_container.pack(pady=10, fill="both", expand=True)

        # Canvas para scroll
        self.chat_canvas = Canvas(self.chat_frame_container, bg=self.bg_color, highlightthickness=0)
        self.chat_canvas.pack(side="left", fill="both", expand=True)

        # Barra de desplazamiento
        self.scrollbar = Scrollbar(self.chat_frame_container, command=self.chat_canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Frame interno dentro del Canvas para los mensajes
        self.chat_frame = Frame(self.chat_canvas, bg=self.bg_color)
        self.chat_window = self.chat_canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")

        # Vincular el scroll con el Canvas
        self.chat_frame.bind("<Configure>", lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.bind("<Configure>", self.resize_canvas)

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
            fg="#000000",  # Texto negro en el botón
            activebackground="#ffc107",  # Amarillo oscuro al presionar
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
            self.display_message(response, "bot")  # Mostrar respuesta del bot
        self.user_input.set("")  # Limpiar campo de entrada

    def display_message(self, message, sender):
        # Crear burbuja con bordes redondeados
        bubble_color = self.user_bg_color if sender == "user" else self.bot_bg_color
        text_color = "#ffffff"  # Texto blanco
        anchor = "e" if sender == "user" else "w"  # Alinear derecha para usuario, izquierda para bot

        # Crear Frame para la burbuja
        bubble_frame = Frame(self.chat_frame, bg=self.bg_color)
        bubble_frame.pack(fill="x", padx=10, pady=5)  # Expandir al ancho del chat

        # Crear etiqueta con el mensaje
        label = Label(
            bubble_frame,
            text=message,
            wraplength=self.chat_canvas.winfo_width() - 50,  # Limitar ancho relativo al tamaño del Canvas
            justify=LEFT if sender == "bot" else RIGHT,
            font=self.font,
            bg=bubble_color,
            fg=text_color,
            padx=10,
            pady=5
        )
        # Alinear etiqueta dentro del Frame
        # Alinear etiqueta dentro del Frame
        if sender == "user":
            label.pack(side="right", anchor="e", padx=5)  # Usuario a la derecha
        else:
            label.pack(side="left", anchor="w", padx=5)  # Bot a la izquierda

        # Auto-scroll hacia abajo
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1)

    def resize_canvas(self, event):
        # Ajustar el ancho del Frame interno al tamaño del Canvas
        canvas_width = event.width
        self.chat_canvas.itemconfig(self.chat_window, width=canvas_width)


# Ejecutar la aplicación
def main():
    root = Tk()
    app = ChatBotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
