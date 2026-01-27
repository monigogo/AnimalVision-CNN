let model;
let isDetecting = false;
let animationId;

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const predictionEl = document.getElementById('prediction');
const emojiEl = document.getElementById('emoji');
const confidenceEl = document.getElementById('confidence');
const loadingEl = document.getElementById('loading');

// Cargar modelo
window.addEventListener('load', async () => {
    loadingEl.classList.add('active');
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log('✅ Modelo cargado');
        loadingEl.classList.remove('active');
        startBtn.disabled = false;
    } catch (error) {
        console.error('❌ Error:', error);
        predictionEl.textContent = 'Error al cargar el modelo';
        loadingEl.classList.remove('active');
    }
});

// Iniciar cámara
startBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' }
        });
        video.srcObject = stream;
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isDetecting = true;
        
        video.addEventListener('loadeddata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            detectFrame();
        });
        
    } catch (error) {
        console.error('❌ Error con cámara:', error);
        alert('No se pudo acceder a la cámara');
    }
});

// Detener
stopBtn.addEventListener('click', () => {
    isDetecting = false;
    if (animationId) cancelAnimationFrame(animationId);
    
    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach(track => track.stop());
    
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    predictionEl.textContent = 'Cámara detenida';
    emojiEl.textContent = '📷';
    confidenceEl.innerHTML = '';
});

// Detección continua
async function detectFrame() {
    if (!isDetecting) return;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    await predict();
    animationId = requestAnimationFrame(detectFrame);
}

// Predicción
async function predict() {
    try {
        const img = tf.browser.fromPixels(canvas)
            .resizeBilinear([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims(0);
        
        const prediction = await model.predict(img).data();
        const dogProb = prediction[0];
        
        let label, confidence, emoji, color;
        
        if (dogProb > 0.5) {
            label = 'Perro';
            confidence = (dogProb * 100).toFixed(1);
            emoji = '🐶';
            color = '#4CAF50';
        } else {
            label = 'Gato';
            confidence = ((1 - dogProb) * 100).toFixed(1);
            emoji = '🐱';
            color = '#FF9800';
        }
        
        predictionEl.textContent = label;
        emojiEl.textContent = emoji;
        confidenceEl.innerHTML = `
            <div style="margin-top: 10px;">
                Confianza: ${confidence}%
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%; background: ${color};"></div>
            </div>
        `;
        
        img.dispose();
        
    } catch (error) {
        console.error('❌ Error en predicción:', error);
    }
}