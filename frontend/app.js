// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let recentTransactions = [];
let confidenceChart, categoryChart, timelineChart;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeCharts();
    initializeForms();
    loadTaxonomy();
    loadStats();
    loadMetrics();

    // Refresh data every 30 seconds
    setInterval(loadStats, 30000);
});

// Navigation
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('href').substring(1);

            // Update active nav
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show section
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.getElementById(target).classList.add('active');
        });
    });
}

// Initialize Charts
function initializeCharts() {
    // Confidence Distribution Chart
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'doughnut',
        data: {
            labels: ['High (≥85%)', 'Medium (70-85%)', 'Low (<70%)'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Category Distribution Chart
    const categoryCtx = document.getElementById('categoryChart').getContext('2d');
    categoryChart = new Chart(categoryCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Transactions',
                data: [],
                backgroundColor: '#6366f1',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Timeline Chart
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    timelineChart = new Chart(timelineCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Avg Latency (ms)',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Latency (ms)'
                    }
                }
            }
        }
    });
}

// Initialize Forms
function initializeForms() {
    // Single transaction form
    document.getElementById('single-transaction-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await categorizeSingleTransaction();
    });

    // File upload
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');

    fileInput.addEventListener('change', handleFileUpload);

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#6366f1';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#e5e7eb';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#e5e7eb';
        const file = e.dataTransfer.files[0];
        if (file) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload();
        }
    });
}

// Categorize Single Transaction
async function categorizeSingleTransaction() {
    showLoading();

    const transaction = {
        transaction_id: document.getElementById('txn-id').value,
        merchant_raw: document.getElementById('merchant-raw').value,
        amount: parseFloat(document.getElementById('amount').value),
        currency: document.getElementById('currency').value,
        timestamp: new Date().toISOString(),
        channel: document.getElementById('channel').value,
        location: document.getElementById('location').value || null,
        mcc_code: document.getElementById('mcc-code').value || null
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/categorize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transactions: [transaction]
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data.results[0]);
            addToRecentTransactions(data.results[0]);
            updateStats();
        } else {
            alert('Error: ' + (data.detail || 'Failed to categorize transaction'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to API. Make sure the server is running at ' + API_BASE_URL);
    } finally {
        hideLoading();
    }
}

// Display Result
function displayResult(result) {
    const container = document.getElementById('result-container');

    const confidenceClass = result.confidence >= 0.85 ? 'high' :
                           result.confidence >= 0.70 ? 'medium' : 'low';

    const confidenceLabel = result.confidence >= 0.85 ? 'High Confidence' :
                           result.confidence >= 0.70 ? 'Medium Confidence' : 'Low Confidence';

    container.innerHTML = `
        <div class="result-box">
            <div class="result-header">
                <h4>Categorization Result</h4>
                <span class="confidence-badge ${confidenceClass}">
                    ${confidenceLabel}: ${(result.confidence * 100).toFixed(1)}%
                </span>
            </div>

            <div class="category-hierarchy">
                <div class="category-level">
                    <strong>L1:</strong> ${result.category.l1}
                </div>
                <div class="category-level">
                    <strong>L2:</strong> ${result.category.l2}
                </div>
                <div class="category-level">
                    <strong>L3:</strong> ${result.category.l3}
                </div>
            </div>

            <div class="result-details">
                <div class="detail-item">
                    <span>Processing Time:</span>
                    <strong>${result.processing_time_ms.toFixed(2)}ms</strong>
                </div>
                <div class="detail-item">
                    <span>Transaction ID:</span>
                    <strong>${result.transaction_id}</strong>
                </div>
                ${result.explanation && result.explanation.merchant_cleaned ? `
                <div class="detail-item">
                    <span>Cleaned Merchant:</span>
                    <strong>${result.explanation.merchant_cleaned}</strong>
                </div>
                ` : ''}
            </div>

            ${result.confidence < 0.70 ? `
                <div style="margin-top: 1rem; padding: 1rem; background: #fee2e2; border-radius: 6px; color: #991b1b;">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Low Confidence:</strong> This transaction should be reviewed manually.
                </div>
            ` : ''}
        </div>
    `;
}

// Add to Recent Transactions
function addToRecentTransactions(result) {
    recentTransactions.unshift(result);
    if (recentTransactions.length > 10) {
        recentTransactions = recentTransactions.slice(0, 10);
    }

    updateRecentTransactionsList();
}

// Update Recent Transactions List
function updateRecentTransactionsList() {
    const container = document.getElementById('recent-transactions');

    if (recentTransactions.length === 0) {
        container.innerHTML = '<p class="empty-state">No transactions processed yet</p>';
        return;
    }

    container.innerHTML = recentTransactions.map(txn => `
        <div class="transaction-item">
            <div>
                <div class="transaction-merchant">${txn.transaction_id}</div>
                <div class="transaction-category">${txn.category.l3}</div>
            </div>
            <span class="confidence-badge ${txn.confidence >= 0.85 ? 'high' : txn.confidence >= 0.70 ? 'medium' : 'low'}">
                ${(txn.confidence * 100).toFixed(0)}%
            </span>
        </div>
    `).join('');
}

// Handle File Upload
async function handleFileUpload() {
    const file = document.getElementById('file-input').files[0];
    if (!file) return;

    showLoading();

    // Note: This is a simplified version. In production, you'd parse CSV
    // and send transactions in batches to the API

    alert('Batch upload feature requires CSV parsing. For now, use the single transaction form or call the API directly.');

    hideLoading();
}

// Load Taxonomy
async function loadTaxonomy() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/taxonomy`);
        const taxonomy = await response.json();

        displayTaxonomy(taxonomy);
    } catch (error) {
        console.error('Error loading taxonomy:', error);
    }
}

// Display Taxonomy
function displayTaxonomy(taxonomy) {
    const container = document.getElementById('taxonomy-tree');

    container.innerHTML = taxonomy.categories.map(l1 => `
        <div class="taxonomy-l1">
            <div class="taxonomy-l1-title">
                ${l1.l1} (${l1.l1_id})
            </div>
            <div class="taxonomy-l2-list">
                ${l1.l2_subcategories.map(l2 => `
                    <div class="taxonomy-l2">
                        <div class="taxonomy-l2-title">
                            ${l2.l2} (${l2.l2_id})
                        </div>
                        <div class="taxonomy-l3-list">
                            ${l2.l3_types.map(l3 => `
                                <div class="taxonomy-l3-item">
                                    ${l3.l3}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

// Load Stats
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/stats`);
        const stats = await response.json();

        // Update dashboard metrics (mock data for now)
        document.getElementById('accuracy-metric').textContent = '≥90%';
        document.getElementById('latency-metric').textContent = '<200ms';
        document.getElementById('confidence-metric').textContent = '--';
        document.getElementById('transactions-metric').textContent = recentTransactions.length.toString();

    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load Metrics
function loadMetrics() {
    // Mock data for demonstration
    updateMetricsDisplay({
        l1_accuracy: 0.95,
        l2_accuracy: 0.92,
        l3_accuracy: 0.90,
        f1_score: 0.90,
        avg_latency: 145,
        p95_latency: 185,
        p99_latency: 195,
        high_confidence: 0.75,
        medium_confidence: 0.20,
        low_confidence: 0.05
    });
}

// Update Metrics Display
function updateMetricsDisplay(metrics) {
    document.getElementById('l1-accuracy').textContent = (metrics.l1_accuracy * 100).toFixed(1) + '%';
    document.getElementById('l2-accuracy').textContent = (metrics.l2_accuracy * 100).toFixed(1) + '%';
    document.getElementById('l3-accuracy').textContent = (metrics.l3_accuracy * 100).toFixed(1) + '%';
    document.getElementById('f1-score').textContent = metrics.f1_score.toFixed(2);

    document.getElementById('avg-latency').textContent = metrics.avg_latency.toFixed(0) + ' ms';
    document.getElementById('p95-latency').textContent = metrics.p95_latency.toFixed(0) + ' ms';
    document.getElementById('p99-latency').textContent = metrics.p99_latency.toFixed(0) + ' ms';

    document.getElementById('high-conf').textContent = (metrics.high_confidence * 100).toFixed(0) + '%';
    document.getElementById('medium-conf').textContent = (metrics.medium_confidence * 100).toFixed(0) + '%';
    document.getElementById('low-conf').textContent = (metrics.low_confidence * 100).toFixed(0) + '%';
    document.getElementById('review-queue').textContent = Math.floor(recentTransactions.length * metrics.low_confidence);

    // Update confidence chart
    confidenceChart.data.datasets[0].data = [
        metrics.high_confidence * 100,
        metrics.medium_confidence * 100,
        metrics.low_confidence * 100
    ];
    confidenceChart.update();
}

// Update Stats
function updateStats() {
    // Update category distribution
    const categoryCounts = {};
    recentTransactions.forEach(txn => {
        const l1 = txn.category.l1;
        categoryCounts[l1] = (categoryCounts[l1] || 0) + 1;
    });

    categoryChart.data.labels = Object.keys(categoryCounts);
    categoryChart.data.datasets[0].data = Object.values(categoryCounts);
    categoryChart.update();

    // Update timeline (mock data)
    const now = new Date();
    const times = Array.from({length: 10}, (_, i) => {
        const d = new Date(now - (9 - i) * 60000);
        return d.toLocaleTimeString();
    });

    timelineChart.data.labels = times;
    timelineChart.data.datasets[0].data = recentTransactions.slice(0, 10).map(t => t.processing_time_ms).reverse();
    timelineChart.update();
}

// Loading Overlay
function showLoading() {
    document.getElementById('loading-overlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}
