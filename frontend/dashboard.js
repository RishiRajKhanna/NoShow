document.addEventListener('DOMContentLoaded', () => {
    const datePicker = document.getElementById('date-picker');
    const loadDataBtn = document.getElementById('load-data-btn');
    const summaryCardsContainer = document.getElementById('summary-cards');
    const appointmentTableBody = document.getElementById('appointment-table-body');

    // Set default date to today
    const today = new Date().toISOString().split('T')[0];
    datePicker.value = today;

    loadDataBtn.addEventListener('click', () => {
        const selectedDate = datePicker.value;
        if (selectedDate) {
            fetchAndDisplayData(selectedDate);
        }
    });

    async function fetchAndDisplayData(date) {
        // Show loading state
        loadDataBtn.disabled = true;
        loadDataBtn.textContent = 'Loading...';
        summaryCardsContainer.innerHTML = '';
        appointmentTableBody.innerHTML = '<tr><td colspan="7">Loading appointments...</td></tr>';

        try {
            const response = await fetch(`http://127.0.0.1:8000/get_appointments_by_date/?date=${date}`);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            updateSummaryCards(data.summary);
            updateAppointmentTable(data.appointments);

        } catch (error) {
            console.error('Error fetching data:', error);
            summaryCardsContainer.innerHTML = '<p>Could not load summary data.</p>';
            appointmentTableBody.innerHTML = `<tr><td colspan="7">Error: ${error.message}</td></tr>`;
        } finally {
            // Restore button state
            loadDataBtn.disabled = false;
            loadDataBtn.textContent = 'Load Data';
        }
    }

    function updateSummaryCards(summary) {
        summaryCardsContainer.innerHTML = `
            <div class="card">
                <h3>Total Appointments Today</h3>
                <p>${summary.total_appointments}</p>
            </div>
            <div class="card">
                <h3>Predicted No-Shows</h3>
                <p>${summary.predicted_noshows}</p>
            </div>
            <div class="card">
                <h3>No-Show Rate</h3>
                <p>${summary.noshow_rate}%</p>
            </div>
            <div class="card">
                <h3>Top Risk Factors</h3>
                <p style="font-size: 1.2rem;">${summary.top_risk_factors}</p>
            </div>
        `;
    }

    function updateAppointmentTable(appointments) {
        if (appointments.length === 0) {
            appointmentTableBody.innerHTML = '<tr><td colspan="7">No appointments found for this date.</td></tr>';
            return;
        }

        appointmentTableBody.innerHTML = ''; // Clear loading message

        appointments.forEach(appt => {
            const row = document.createElement('tr');
            
            const predictionClass = getPredictionClass(appt.prediction);

            row.innerHTML = `
                <td>${appt.id}</td>
                <td>${appt.patient_name}</td>
                <td>${appt.time}</td>
                <td>${appt.reason}</td>
                <td>${appt.risk_factors}</td>
                <td class="${predictionClass}">${appt.prediction}</td>
                <td>${appt.action}</td>
            `;
            appointmentTableBody.appendChild(row);
        });
    }

    function getPredictionClass(prediction) {
        if (prediction === 'High Risk') {
            return 'prediction-high';
        }
        if (prediction === 'Medium Risk') {
            return 'prediction-medium';
        }
        return 'prediction-low';
    }

    // Automatically load data for the default date on page load
    fetchAndDisplayData(today);
});
