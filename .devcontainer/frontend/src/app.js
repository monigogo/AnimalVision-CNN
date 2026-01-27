let isDetecting = false;
let animationId;
let isProcessing = false;
let lastPredictionTime = 0;
const PREDICTION_INTERVAL = 3000; // 3 segundos entre predicciones para estabilidad

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Establecer tamaño inicial del canvas
canvas.width = 640;
canvas.height = 480;
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const imageUpload = document.getElementById('imageUpload');
const predictionEl = document.getElementById('prediction');
const emojiEl = document.getElementById('emoji');
const confidenceEl = document.getElementById('confidence');
const loadingEl = document.getElementById('loading');

// No cargamos modelo en frontend
window.addEventListener('load', () => {
    loadingEl.classList.remove('active');
    startBtn.disabled = false;
});

// Iniciar cámara
startBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            }
        });
        video.srcObject = stream;

        video.style.display = 'block';
        canvas.style.display = 'none';

        startBtn.disabled = true;
        stopBtn.disabled = false;
        isDetecting = true;

        video.addEventListener('loadeddata', () => {
            canvas.width = Math.min(video.videoWidth, 640);
            canvas.height = Math.min(video.videoHeight, 480);
            lastPredictionTime = Date.now();
            detectFrame();
        });

    } catch (error) {
        alert('No se pudo acceder a la cámara');
    }
});

// Detener cámara
stopBtn.addEventListener('click', () => {
    isDetecting = false;
    if (animationId) cancelAnimationFrame(animationId);

    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach(track => track.stop());

    video.srcObject = null;
    video.style.display = 'block';
    canvas.style.display = 'none';

    startBtn.disabled = false;
    stopBtn.disabled = true;
    predictionEl.textContent = 'Cámara detenida';
    emojiEl.textContent = '📷';
    confidenceEl.innerHTML = '';
});

// Cargar imagen desde archivo
imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (isDetecting) {
        isDetecting = false;
        if (animationId) cancelAnimationFrame(animationId);
        const stream = video.srcObject;
        if (stream) stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }

    video.style.display = 'none';
    canvas.style.display = 'block';

    predictionEl.textContent = 'Analizando imagen...';
    emojiEl.textContent = '🔍';
    confidenceEl.innerHTML = '';

    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
            canvas.width = 640;
            canvas.height = 480;

            const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
            const x = (canvas.width / 2) - (img.width / 2) * scale;
            const y = (canvas.height / 2) - (img.height / 2) * scale;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

            await predictImage(file);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

// Detección continua (video → backend) con throttle real
async function detectFrame() {
    if (!isDetecting) return;

    const now = Date.now();
    
    // Solo predecir si han pasado 3 segundos desde la última predicción
    if (!isProcessing && (now - lastPredictionTime) >= PREDICTION_INTERVAL) {
        lastPredictionTime = now;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
            if (blob) {
                await predictFrame(blob);
            }
        }, "image/jpeg", 1.0); // Máxima calidad sin compresión
    }

    animationId = requestAnimationFrame(detectFrame);
}

// Predicción de imagen cargada
async function predictImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        showResult(data);

    } catch (error) {
        predictionEl.textContent = "Error al procesar la imagen";
        emojiEl.textContent = "⚠️";
    }
}

// Predicción de frame de video
async function predictFrame(blob) {
    isProcessing = true; // Marcar como ocupado
    
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    formData.append("source", "video"); // Indicar que es desde video

    try {
        const res = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        console.log('Respuesta API:', data); // Debug
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        showResult(data);

    } catch (error) {
        console.error('Error en predicción:', error);
        predictionEl.textContent = "Error: " + error.message;
        emojiEl.textContent = "⚠️";
        confidenceEl.innerHTML = '';
    } finally {
        isProcessing = false; // Liberar para la siguiente predicción
    }
}

// Mostrar resultado
function showResult(data) {
    console.log('Mostrando resultado:', data); // Debug
    
    if (!data || !data.label) {
        predictionEl.textContent = "Respuesta inválida";
        emojiEl.textContent = "⚠️";
        confidenceEl.innerHTML = '';
        return;
    }
    
    predictionEl.textContent = data.label;

    emojiEl.textContent =
        data.label === "perro" ? "🐶" :
        data.label === "gato" ? "🐱" : 
        data.label === "no identificado" ? "❓" : "⚠️";

    confidenceEl.innerHTML = `Confianza: ${data.confidence.toFixed(1)}%`;
}