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

// Funciones de procesamiento de texto
function tokenize(sentence) {
    return sentence.toLowerCase().match(/\b(\w+)\b/g);
}


function levenshteinDistance(a, b) {
    const matrix = Array.from({ length: a.length + 1 }, (_, i) =>
        Array.from({ length: b.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
    );

    for (let i = 1; i <= a.length; i++) {
        for (let j = 1; j <= b.length; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            );
        }
    }

    return matrix[a.length][b.length];
}

function similarity(word1, word2) {
    const maxLength = Math.max(word1.length, word2.length);
    if (maxLength === 0) return 1;
    const distance = levenshteinDistance(word1, word2);
    return 1 - distance / maxLength;
}

// Bag of Words mejorado
function bagOfWords(sentence, words) {
    const sentenceTokens = tokenize(sentence);
    const bag = Array(words.length).fill(0);

    sentenceTokens.forEach(token => {
        let maxSimilarity = 0;
        let bestMatchIndex = -1;

        words.forEach((word, index) => {
            const sim = similarity(token, word);
            if (sim > maxSimilarity) {
                maxSimilarity = sim;
                bestMatchIndex = index;
            }
        });

        if (maxSimilarity > 0.8 && bestMatchIndex !== -1) {
            bag[bestMatchIndex] = 1;
        }
    });

    return bag;
}

// Predicción de clase
async function predictClass(sentence) {
    const inputBag = bagOfWords(sentence, words);
    const inputTensor = tf.tensor([inputBag]);

    const predictions = await model.predict(inputTensor).array();
    const maxIdx = predictions[0].indexOf(Math.max(...predictions[0]));

    return predictions[0][maxIdx] > 0.3 ? classes[maxIdx] : null;
}

// Obtener respuesta
function getResponse(intentTag) {
    const intent = intents.intents.find(i => i.tag === intentTag);
    return intent ? intent.responses[Math.floor(Math.random() * intent.responses.length)] : "No entiendo lo que dices.";
}

function formatResponse(response) {
    return response.replace(/\n/g, '<br>');
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

    const intent = await predictClass(userInput);
    const response = formatResponse(getResponse(intent));

    messages.removeChild(botTyping);
    messages.innerHTML += `<div class="message bot-message">${response}</div>`;
    messages.scrollTop = messages.scrollHeight;

    document.getElementById('user-input').value = '';
});

// Cargar recursos al inicio
loadResources();