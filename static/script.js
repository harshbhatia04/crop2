const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultCard = document.getElementById('result-card');
const idleState = document.getElementById('idle');
const loadingState = document.getElementById('loading');
const resultsArea = document.getElementById('results');
const resetBtn = document.getElementById('reset-btn');

// Modal Elements
const expertModal = document.getElementById('expert-modal');
const openExpertBtn = document.getElementById('open-expert-btn');
const closeExpertBtn = document.getElementById('close-expert-btn');
const downloadPdfBtn = document.getElementById('download-pdf-btn');
const listenBtn = document.getElementById('listen-btn');
const aiAdviceText = document.getElementById('res-ai-advice');
const aiLoading = document.getElementById('ai-loading');
const langSelect = document.getElementById('lang-select');
const refreshAiBtn = document.getElementById('refresh-ai-btn');

let currentPrediction = null;
let currentAudio = null;

// Drag and drop mechanics
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleUpload(e.target.files[0]);
});

function handleUpload(file) {
    resultCard.classList.remove('disabled');
    idleState.classList.add('hidden');
    resultsArea.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const previewImg = document.getElementById('upload-preview');
    const placeholder = document.getElementById('upload-placeholder');
    previewImg.src = URL.createObjectURL(file);
    previewImg.classList.remove('hidden');
    placeholder.classList.add('hidden');

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => {
        currentPrediction = data;
        document.getElementById('res-disease').innerText = data.disease.toUpperCase();
        document.getElementById('res-crop').innerText = `${data.crop} Crop`;
        document.getElementById('conf-val').innerText = `${data.confidence} MATCH`;
        
        const dangerPill = document.getElementById('danger-val');
        dangerPill.innerText = data.danger_level;
        const colors = { 'CRITICAL': '#ff4757', 'HIGH': '#ffa502', 'MODERATE': '#3742fa', 'LOW': '#2ed573', 'HEALTHY': '#2f3542' };
        dangerPill.style.backgroundColor = colors[data.danger_level] || '#1a1a1a';

        if (data.visual_explanation) {
            const cleanPath = data.visual_explanation.replace(/\\/g, '/');
            document.getElementById('gradcam-img').src = `${cleanPath}?t=${new Date().getTime()}`;
        }

        document.getElementById('res-organic').innerText = data.organic_treatment;
        document.getElementById('res-chemical').innerText = data.chemical_treatment;

        loadingState.classList.add('hidden');
        resultsArea.classList.remove('hidden');
    })
    .catch(err => {
        console.error(err);
        alert("Server Error!");
        loadingState.classList.add('hidden');
        idleState.classList.remove('hidden');
    });
}

// Modal Toggle Logic
openExpertBtn.addEventListener('click', () => {
    expertModal.classList.remove('hidden');
    if (currentPrediction) {
        getAiAdvice(currentPrediction.crop, currentPrediction.disease, langSelect.value);
    }
});

closeExpertBtn.addEventListener('click', () => {
    expertModal.classList.add('hidden');
    stopSpeaking();
});

refreshAiBtn.addEventListener('click', () => {
    if (currentPrediction) {
        getAiAdvice(currentPrediction.crop, currentPrediction.disease, langSelect.value);
    }
});

// Sarvam AI Voice Logic
listenBtn.addEventListener('click', async () => {
    const text = aiAdviceText.innerText;
    const lang = langSelect.value;
    
    stopSpeaking();
    listenBtn.innerText = "🔊 LOADING SARVAM AI...";
    listenBtn.classList.add('btn-orange');

    try {
        const response = await fetch("/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, language: lang })
        });
        const data = await response.json();
        
        if (data.audio) {
            const audioBlob = b64toBlob(data.audio, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            currentAudio = new Audio(audioUrl);
            currentAudio.play();
            listenBtn.innerText = "🔊 SPEAKING (SARVAM)...";
            currentAudio.onended = () => stopSpeaking();
        } else {
            throw new Error("No audio returned");
        }
    } catch (err) {
        console.error("Sarvam Error, falling back:", err);
        fallbackToBrowser(text, lang);
    }
});

function b64toBlob(b64Data, contentType) {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) byteNumbers[i] = slice.charCodeAt(i);
        byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, {type: contentType});
}

function fallbackToBrowser(text, lang) {
    const utterance = new SpeechSynthesisUtterance(text);
    let targetLang = lang === "Hindi" ? 'hi-IN' : 'en-US';
    utterance.lang = targetLang;
    window.speechSynthesis.speak(utterance);
    listenBtn.innerText = "🔊 SPEAKING (BROWSER)...";
    utterance.onend = () => stopSpeaking();
}

function stopSpeaking() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    window.speechSynthesis.cancel();
    listenBtn.innerText = "🔊 LISTEN TO EXPERT";
    listenBtn.classList.remove('btn-orange');
}

window.addEventListener('click', (e) => {
    if (e.target === expertModal) {
        expertModal.classList.add('hidden');
        stopSpeaking();
    }
});

async function getAiAdvice(crop, disease, language) {
    aiAdviceText.classList.add('hidden');
    aiLoading.classList.remove('hidden');
    
    try {
        const response = await fetch("/ask_ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ crop, disease, language })
        });
        
        const data = await response.json();
        if (!response.ok) {
            aiAdviceText.innerText = `Error: ${data.advice || response.statusText}`;
        } else {
            aiAdviceText.innerText = data.advice;
        }
    } catch (err) {
        aiAdviceText.innerText = `Network Error: ${err.message}. Ensure backend is running!`;
    } finally {
        aiLoading.classList.add('hidden');
        aiAdviceText.classList.remove('hidden');
    }
}

// PDF Generation
downloadPdfBtn.addEventListener('click', async () => {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const disease = document.getElementById('res-disease').innerText;
    const aiAdvice = aiAdviceText.innerText;
    doc.text(`Expert Diagnosis: ${disease}`, 15, 25);
    doc.text(doc.splitTextToSize(aiAdvice, 180), 15, 40);
    doc.save(`Expert_Report_${disease}.pdf`);
});

resetBtn.addEventListener('click', () => {
    resultsArea.classList.add('hidden');
    idleState.classList.remove('hidden');
    resultCard.classList.add('disabled');
    document.getElementById('upload-preview').classList.add('hidden');
    document.getElementById('upload-placeholder').classList.remove('hidden');
    fileInput.value = "";
    aiAdviceText.innerText = "Click 'Get Solution' to generate AI advice.";
    langSelect.value = "English";
    stopSpeaking();
});
