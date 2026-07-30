const form = document.getElementById('predict-form');
const textInput = document.getElementById('text-input');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const fileDropZone = document.getElementById('file-drop-zone');
const button = document.getElementById('predict-button');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const errorBox = document.getElementById('error');

const hideMessages = () => [loading, result, errorBox].forEach((item) => item.classList.remove('visible'));

const updateSelectedFile = () => {
    fileName.textContent = fileInput.files[0]?.name || 'No media selected';
};

fileInput.addEventListener('change', updateSelectedFile);

['dragenter', 'dragover'].forEach((eventName) => {
    fileDropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        fileDropZone.classList.add('is-dragging');
    });
});

['dragleave', 'drop'].forEach((eventName) => {
    fileDropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        fileDropZone.classList.remove('is-dragging');
    });
});

fileDropZone.addEventListener('drop', (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) return;

    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    updateSelectedFile();
});

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const text = textInput.value.trim();
    const file = fileInput.files[0];
    hideMessages();

    if (!text && !file) {
        errorBox.classList.add('visible');
        document.getElementById('error-message').textContent = 'Enter text or choose an audio/video file before running classification.';
        return;
    }

    const formData = new FormData();
    if (text) formData.append('text', text);
    if (file) formData.append('file', file);

    loading.classList.add('visible');
    button.disabled = true;

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'The server could not complete the request.');
        }

        document.getElementById('label').textContent = data.label;
        document.getElementById('confidence').textContent = data.confidence;
        result.classList.add('visible');
    } catch (error) {
        document.getElementById('error-message').textContent = error.message || 'Failed to connect to the server.';
        errorBox.classList.add('visible');
    } finally {
        loading.classList.remove('visible');
        button.disabled = false;
    }
});
