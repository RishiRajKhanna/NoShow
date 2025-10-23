
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictionText = document.getElementById('prediction-text');
    const probabilityText = document.getElementById('probability-text');

    // Set default date to today
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('appointment_date').value = today;

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent the default form submission (page reload)

        // 1. Collect data from the form
        const formData = new FormData(form);
        const data = {
            appointment_date: formData.get('appointment_date'),
            appointment_time: formData.get('appointment_time'),
            past_noshow_rate: parseFloat(formData.get('past_noshow_rate')),
            days_since_last_appt: parseInt(formData.get('days_since_last_appt')),
            duration_min: parseInt(formData.get('duration_min')),
            appointment_type: formData.get('appointment_type'),
            doctor: formData.get('doctor'),
        };

        // Basic validation
        if (!data.appointment_time) {
            alert('Please enter an appointment time.');
            return;
        }

        // Show loading state (optional)
        resultContainer.classList.add('hidden');
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.textContent = 'Predicting...';

        try {
            // 2. Send data to the backend API
            const response = await fetch('http://127.0.0.1:8000/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // 3. Display the result
            displayResult(result);

        } catch (error) {
            console.error('Error making prediction:', error);
            alert('Could not connect to the prediction service. Please ensure the backend server is running.');
        } finally {
            // Restore button state
            submitButton.disabled = false;
            submitButton.textContent = 'Predict No-Show Risk';
        }
    });

    function displayResult(result) {
        resultContainer.classList.remove('hidden');
        const probabilityPercent = Math.round(result.no_show_probability * 100);

        if (result.prediction === 'No-Show') {
            predictionText.textContent = 'Prediction: HIGH RISK of No-Show';
            probabilityText.textContent = `Confidence: ${probabilityPercent}% probability of no-show.`;
            predictionText.className = 'no-show-result';
            resultContainer.className = 'no-show-result';
        } else {
            predictionText.textContent = 'Prediction: Low risk of no-show';
            probabilityText.textContent = `Confidence: ${probabilityPercent}% probability of no-show.`;
            predictionText.className = 'show-result';
            resultContainer.className = 'show-result';
        }
    }
});
