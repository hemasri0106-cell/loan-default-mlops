/**
 * CreditAI FinTech Dashboard JavaScript
 * Handles AJAX prediction calls, DOM updates, sample profiles, and error handling.
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnSpinner = submitBtn.querySelector('.btn-spinner');
    const fillSampleBtn = document.getElementById('fill-sample-btn');

    const resultCard = document.getElementById('result-card');
    const emptyResultCard = document.getElementById('empty-result-card');
    const errorCard = document.getElementById('error-card');
    const errorMessage = document.getElementById('error-message');

    // Sample Profile Preset for quick testing
    const sampleProfile = {
        loan_amnt: 15000,
        term: " 36 months",
        int_rate: 10.99,
        installment: 490.93,
        purpose: "debt_consolidation",
        issue_d: "2018-01-01",
        annual_inc: 85000,
        emp_length: "10+ years",
        home_ownership: "MORTGAGE",
        addr_state: "CA",
        fico_range_low: 720,
        fico_range_high: 724,
        grade: "B",
        sub_grade: "B1",
        dti: 12.5,
        revol_util: 32.4,
        open_acc: 10,
        verification_status: "Source Verified"
    };

    // Load sample profile click listener
    if (fillSampleBtn) {
        fillSampleBtn.addEventListener('click', () => {
            Object.keys(sampleProfile).forEach(key => {
                const input = form.querySelector(`[name="${key}"]`);
                if (input) {
                    input.value = sampleProfile[key];
                }
            });
        });
    }

    // Form Submission Listener
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 1. UI Loading State
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        btnSpinner.classList.remove('hidden');
        
        errorCard.classList.add('hidden');

        // 2. Build Payload
        const formData = new FormData(form);
        const payload = {};

        const numericFields = [
            'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'open_acc', 'revol_util'
        ];

        formData.forEach((value, key) => {
            if (numericFields.includes(key)) {
                payload[key] = parseFloat(value) || 0.0;
            } else {
                payload[key] = value.trim();
            }
        });

        try {
            // 3. Execute POST Request to FastAPI /predict
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || data.detail || 'Prediction failed.');
            }

            // 4. Update Result Card DOM
            displayPredictionResults(data);

        } catch (err) {
            console.error('Prediction Error:', err);
            errorMessage.textContent = err.message || 'Unable to connect to prediction service.';
            errorCard.classList.remove('hidden');
        } finally {
            // Reset Button State
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            btnSpinner.classList.add('hidden');
        }
    });

    function displayPredictionResults(data) {
        emptyResultCard.classList.add('hidden');
        resultCard.classList.remove('hidden');

        const timestampEl = document.getElementById('result-timestamp');
        const badgeCardEl = document.getElementById('risk-badge-container');
        const badgeEl = document.getElementById('risk-badge');
        const headlineEl = document.getElementById('prediction-text');
        const subtextEl = document.getElementById('prediction-desc');
        
        const probValEl = document.getElementById('probability-value');
        const progressBarEl = document.getElementById('progress-bar');
        const riskScoreEl = document.getElementById('risk-score-value');
        const latencyEl = document.getElementById('latency-value');

        // Set Values
        timestampEl.textContent = data.timestamp || new Date().toLocaleTimeString();
        headlineEl.textContent = data.prediction;
        badgeEl.textContent = data.prediction_label;
        
        const probPct = (data.probability * 100).toFixed(1);
        probValEl.textContent = `${probPct}%`;
        riskScoreEl.textContent = `${data.risk_score} / 100`;
        latencyEl.textContent = `${data.latency_ms} ms`;

        // Style Risk Card based on prediction
        badgeCardEl.classList.remove('low-risk', 'high-risk');
        
        if (data.prediction_label === 'Low Risk') {
            badgeCardEl.classList.add('low-risk');
            subtextEl.textContent = 'Statistical default probability is within safe underwriting boundaries.';
            progressBarEl.style.width = `${Math.max(probPct, 6)}%`;
        } else {
            badgeCardEl.classList.add('high-risk');
            subtextEl.textContent = 'Higher probability of credit default detected. Secondary manual review recommended.';
            progressBarEl.style.width = `${Math.min(probPct, 100)}%`;
        }

        // Smooth scroll to result on mobile views
        if (window.innerWidth < 1024) {
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
});
