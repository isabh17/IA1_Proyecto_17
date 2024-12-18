let model;
let words = [];
let classes = [];
let intents = [];

// Cargar recursos al iniciar
async function loadResources() {
    try {
        model = await tf.loadLayersModel('./model/model.json');

        const wordsResponse = await fetch('./words.json');
        words = await wordsResponse.json();

        const classesResponse = await fetch('./classes.json');
        classes = await classesResponse.json();

        const intentsResponse = await fetch('./intents2.json');
        intents = await intentsResponse.json();

        console.log("Recursos cargados correctamente");
    } catch (error) {
        console.error("Error al cargar los recursos:", error);
        alert("Hubo un problema al cargar el chatbot. Por favor, intenta recargar la página.");
    }
}

// Tokenizar y procesar la entrada del usuario
function tokenize(sentence) {
    return sentence.toLowerCase().match(/\b(\w+)\b/g);
}

function preprocess(sentence) {
    const sentenceTokens = tokenize(sentence);
    return sentenceTokens.map(token => token.toLowerCase()); // Lematización no incluida
}

function bagOfWords(sentence, words) {
    const sentenceTokens = preprocess(sentence);
    const bag = Array(words.length).fill(0);
    sentenceTokens.forEach(token => {
        const index = words.indexOf(token);
        if (index !== -1) {
            bag[index] = 1;
        }
    });
    return bag;
}

// Predecir la clase
const ERROR_THRESHOLD = 0.3; // Ajuste del umbral

async function predictClass(sentence) {
    const inputBag = bagOfWords(sentence, words);
    const inputTensor = tf.tensor([inputBag]);

    const predictions = await model.predict(inputTensor).array();
    console.log("Predicciones:", predictions); // Para depuración

    const predIdx = predictions[0].indexOf(Math.max(...predictions[0]));
    console.log("Índice predicho:", predIdx);

    return predictions[0][predIdx] > ERROR_THRESHOLD ? classes[predIdx] : null;
}

// Obtener una respuesta según la clase
function similarity(str1, str2) {
    if (!str1 || !str2) {
        return 0;
    }

    const tokenize = (text) => {
        if (typeof text !== "string") return [];
        return text.toLowerCase().match(/[a-záéíóúüñ]+/g) || []; // Acepta palabras en español
    };

    const tokens1 = new Set(tokenize(str1));
    const tokens2 = new Set(tokenize(str2));
    const intersection = [...tokens1].filter(token => tokens2.has(token)).length;
    const union = new Set([...tokens1, ...tokens2]).size;

    return union === 0 ? 0 : intersection / union;
}

function getResponse(intentTag, userInput) {
    if (!intentTag) return "Lo siento, no entiendo tu mensaje.";

    const intent = intents.intents.find(i => i.tag === intentTag);
    if (!intent) return "Lo siento, no entiendo tu mensaje.";

    let bestPattern = "";
    let bestScore = 0;

    const cleanInput = userInput.trim().toLowerCase();

    for (const pattern of intent.patterns) {
        console.log(`Comparando: pattern="${pattern}" con userInput="${cleanInput}"`);
        const score = similarity(pattern.toLowerCase(), cleanInput);
        console.log(`Similitud: ${score}`);
        if (score > bestScore) {
            bestScore = score;
            bestPattern = pattern;
        }
    }

    console.log(`Patrón más cercano: ${bestPattern}, Puntuación: ${bestScore}`);

    if (bestScore > 0.1) { // Aumenta el umbral para ser más estricto con las coincidencias
        return intent.responses[Math.floor(Math.random() * intent.responses.length)];
    } else {
        return "Lo siento, no estoy seguro de lo que quieres decir.";
    }
    
}

// Manejar la interacción
document.getElementById('send').addEventListener('click', async () => {
    const userInput = document.getElementById('user-input').value.trim();

    if (!userInput) {
        alert("Por favor, escribe un mensaje.");
        return;
    }

    const messages = document.getElementById('messages');
    messages.innerHTML += `<div class="message user-message">${userInput}</div>`;
    messages.scrollTop = messages.scrollHeight;

    const botTyping = document.createElement("div");
    botTyping.classList.add("message", "bot-message");
    botTyping.textContent = "Bot está escribiendo...";
    messages.appendChild(botTyping);
    messages.scrollTop = messages.scrollHeight;

    try {
        const intent = await predictClass(userInput);
        console.log("Intención predicha:", intent); // Para depuración
    
        // Pasa userInput a la función getResponse
        const response = getResponse(intent, userInput);
        console.log("Respuesta generada:", response); // Para depuración
    
        messages.removeChild(botTyping);
        messages.innerHTML += `<div class="message bot-message">${response}</div>`;
    } catch (error) {
        console.error("Error al procesar el mensaje:", error);
        messages.removeChild(botTyping);
        messages.innerHTML += `<div class="message bot-message">Ocurrió un error. Inténtalo de nuevo más tarde.</div>`;
    }
    
    messages.scrollTop = messages.scrollHeight;
    document.getElementById('user-input').value = '';
});

// Cargar recursos al inicio
loadResources();
