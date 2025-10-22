// Global variables
let currentImagePath = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const downloadBtn = document.getElementById('downloadBtn');

// File upload handlers
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        displayPreview(file);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        fileInput.files = e.dataTransfer.files;
        displayPreview(file);
    }
});

// Display image preview
function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        document.querySelector('.upload-prompt').style.display = 'none';
        analyzeBtn.style.display = 'block';
        results.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    loading.style.display = 'block';
    results.style.display = 'none';
    analyzeBtn.style.display = 'none';

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        currentImagePath = data.image_path;
        displayResults(data);

    } catch (error) {
        alert('Error: ' + error.message);
        analyzeBtn.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
});

// Display results
function displayResults(data) {
    // Risk score
    const riskScore = document.getElementById('riskScore');
    const riskLevel = document.getElementById('riskLevel');

    riskScore.textContent = Math.round(data.risk_score);
    riskLevel.textContent = data.risk_level.toUpperCase();

    // Risk color
    const riskClass = 'risk-' + data.risk_level.toLowerCase().replace(' ', '-');
    riskScore.className = 'risk-score ' + riskClass;
    riskLevel.className = 'risk-level ' + riskClass;

    // Predictions
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = '';

    for (const [className, prob] of Object.entries(data.predictions)) {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-label">${className}</div>
            <div class="prediction-value">${(prob * 100).toFixed(1)}%</div>
        `;
        predictionsDiv.appendChild(item);
    }

    // Visual features
    const featuresDiv = document.getElementById('features');
    featuresDiv.innerHTML = '';

    const featureNames = {
        asymmetry: 'Asymmetry',
        border_irregularity: 'Border Irregularity',
        color_variation: 'Color Variation'
    };

    for (const [key, name] of Object.entries(featureNames)) {
        const value = data.visual_features[key] || 0;
        const item = document.createElement('div');
        item.className = 'feature-item';
        item.innerHTML = `
            <div class="feature-label">
                <span>${name}</span>
                <span>${value.toFixed(2)}</span>
            </div>
            <div class="feature-bar">
                <div class="feature-fill" style="width: ${value * 100}%"></div>
            </div>
        `;
        featuresDiv.appendChild(item);
    }

    // Recommendations
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '';

    data.recommendations.forEach((rec, index) => {
        const item = document.createElement('div');
        let className = 'recommendation';

        if (rec.includes('NOT intended')) {
            className += ' error';
        } else if (rec.includes('HIGH') || rec.includes('URGENT')) {
            className += ' warning';
        } else {
            className += ' info';
        }

        item.className = className;
        item.innerHTML = `<strong>${index + 1}.</strong> ${rec}`;
        recommendationsDiv.appendChild(item);
    });

    // Show results
    results.style.display = 'block';
    analyzeBtn.style.display = 'block';
}

// Download PDF
downloadBtn.addEventListener('click', async () => {
    if (!currentImagePath) return;

    loading.style.display = 'block';

    try {
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_path: currentImagePath })
        });

        if (!response.ok) {
            throw new Error('PDF generation failed');
        }

        // Download file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'lesion_report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        loading.style.display = 'none';
    }
});
