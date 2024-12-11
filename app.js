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

        const intentsResponse = await fetch('./intents.json');
        intents = await intentsResponse.json();

        console.log("Recursos cargados");
    } catch (error) {
        console.error("Error al cargar los recursos:", error);
        alert("Hubo un problema al cargar el chatbot. Por favor, intenta recargar la página.");
    }
}

// Tokenizar y procesar la entrada del usuario
function tokenize(sentence) {
    return sentence.toLowerCase().match(/\b(\w+)\b/g);
}

function bagOfWords(sentence, words) {
    const sentenceTokens = tokenize(sentence);
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
async function predictClass(sentence) {
    const inputBag = bagOfWords(sentence, words);
    const inputTensor = tf.tensor([inputBag]);

    const predictions = await model.predict(inputTensor).array();
    const maxIdx = predictions[0].indexOf(Math.max(...predictions[0]));

    return predictions[0][maxIdx] > 0.25 ? classes[maxIdx] : null;
}

// Obtener una respuesta según la clase
// Contextualizar
function getResponse(intentTag, currentContext) {
    let intent;
    if (currentContext) {
        intent = intents.intents.find(i => i.tag === intentTag && i.context.includes(currentContext));
    }
    if (!intent) {
        intent = intents.intents.find(i => i.tag === intentTag);
    }
    return intent ? intent.responses[Math.floor(Math.random() * intent.responses.length)] : "No entiendo lo que dices.";
}
// Manejar la interacción
document.getElementById('send').addEventListener('click', async () => {
    const userInput = document.getElementById('user-input').value.trim();

    if (!userInput) {
        alert("Por favor, escribe un mensaje.");
        return;
    }

    const messages = document.getElementById('messages');

    // Mostrar el mensaje del usuario
    messages.innerHTML += `<div class="message user-message">${userInput}</div>`;
    messages.scrollTop = messages.scrollHeight;

    // Mostrar indicador de escribiendo
    const botTyping = document.createElement("div");
    botTyping.classList.add("message", "bot-message");
    botTyping.textContent = "Bot está escribiendo...";
    messages.appendChild(botTyping);
    messages.scrollTop = messages.scrollHeight;

    const intent = await predictClass(userInput);
    const response = getResponse(intent);

    // Eliminar indicador y mostrar respuesta del bot
    messages.removeChild(botTyping);
    messages.innerHTML += `<div class="message bot-message">${response}</div>`;
    messages.scrollTop = messages.scrollHeight;

    document.getElementById('user-input').value = '';
});

// Cargar recursos al inicio
loadResources();
