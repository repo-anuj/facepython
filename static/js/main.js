document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const captureBtn = document.getElementById('capture-btn');
    const filterBtns = document.querySelectorAll('.filter-btn');
    const emotionToggle = document.getElementById('emotion-toggle');
    const ageGenderToggle = document.getElementById('age-gender-toggle');
    const mlModeToggle = document.getElementById('ml-mode-toggle');
    const landmarksToggle = document.getElementById('landmarks-toggle');
    const faceRecognitionToggle = document.getElementById('face-recognition-toggle');
    const captureGallery = document.getElementById('capture-gallery');

    // Data collection elements
    const dataCollectionToggle = document.getElementById('data-collection-toggle');
    const collectionForm = document.getElementById('collection-form');
    const personNameInput = document.getElementById('person-name');
    const maxImagesInput = document.getElementById('max-images');
    const startCollectionBtn = document.getElementById('start-collection-btn');
    const stopCollectionBtn = document.getElementById('stop-collection-btn');
    const collectionStatus = document.getElementById('collection-status');
    const collectionProgress = document.getElementById('collection-progress');
    const collectionStatusText = document.getElementById('collection-status-text');
    const collectedPeople = document.getElementById('collected-people');

    // Face recognition training elements
    const trainModelBtn = document.getElementById('train-model-btn');
    const epochsInput = document.getElementById('epochs');
    const batchSizeInput = document.getElementById('batch-size');
    const trainingStatus = document.getElementById('training-status');
    const trainingProgress = document.getElementById('training-progress');
    const trainingStatusText = document.getElementById('training-status-text');
    const modelInfoText = document.getElementById('model-info-text');

    // Emotion recognition training elements
    const trainEmotionBtn = document.getElementById('train-emotion-btn');
    const emotionEpochsInput = document.getElementById('emotion-epochs');
    const emotionBatchSizeInput = document.getElementById('emotion-batch-size');
    const emotionTrainingStatus = document.getElementById('emotion-training-status');
    const emotionTrainingProgress = document.getElementById('emotion-training-progress');
    const emotionTrainingStatusText = document.getElementById('emotion-training-status-text');
    const emotionModelInfoText = document.getElementById('emotion-model-info-text');

    // Capture screenshot
    captureBtn.addEventListener('click', function() {
        fetch('/capture', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addCaptureToGallery(data.filename);
            } else {
                alert('Failed to capture screenshot');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // Filter selection
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const filterId = parseInt(this.dataset.filter);

            // Update UI
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Send to server
            fetch('/toggle_filter', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filter_id: filterId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    alert('Failed to apply filter');
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    });

    // Emotion detection toggle
    emotionToggle.addEventListener('click', function() {
        this.classList.toggle('active');

        fetch('/toggle_emotion', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                this.classList.toggle('active'); // Revert if failed
                alert('Failed to toggle emotion detection');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.classList.toggle('active'); // Revert if error
        });
    });

    // Age/Gender toggle
    ageGenderToggle.addEventListener('click', function() {
        this.classList.toggle('active');

        fetch('/toggle_age_gender', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                this.classList.toggle('active'); // Revert if failed
                alert('Failed to toggle age/gender estimation');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.classList.toggle('active'); // Revert if error
        });
    });

    // ML Mode toggle
    mlModeToggle.addEventListener('click', function() {
        this.classList.toggle('active');

        fetch('/toggle_ml_mode', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                this.classList.toggle('active'); // Revert if failed
                alert('Failed to toggle ML detection mode');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.classList.toggle('active'); // Revert if error
        });
    });

    // Landmarks toggle
    landmarksToggle.addEventListener('click', function() {
        this.classList.toggle('active');

        fetch('/toggle_landmarks', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                this.classList.toggle('active'); // Revert if failed
                alert('Failed to toggle facial landmarks');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.classList.toggle('active'); // Revert if error
        });
    });

    // Face recognition toggle
    faceRecognitionToggle.addEventListener('click', function() {
        this.classList.toggle('active');

        fetch('/toggle_face_recognition', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                this.classList.toggle('active'); // Revert if failed
                alert('Failed to toggle face recognition');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.classList.toggle('active'); // Revert if error
        });
    });

    // Data collection toggle
    dataCollectionToggle.addEventListener('click', function() {
        this.classList.toggle('active');

        fetch('/toggle_data_collection', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (data.data_collection) {
                    collectionForm.style.display = 'block';
                } else {
                    collectionForm.style.display = 'none';
                    collectionStatus.style.display = 'none';
                    stopCollectionBtn.disabled = true;
                    startCollectionBtn.disabled = false;
                }
            } else {
                this.classList.toggle('active'); // Revert if failed
                alert('Failed to toggle data collection mode');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            this.classList.toggle('active'); // Revert if error
        });
    });

    // Start collection button
    startCollectionBtn.addEventListener('click', function() {
        const personName = personNameInput.value.trim();
        if (!personName) {
            alert('Please enter a person name');
            return;
        }

        const maxImages = parseInt(maxImagesInput.value) || 50;

        fetch('/start_collection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                person_name: personName,
                max_images: maxImages
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                collectionStatus.style.display = 'block';
                startCollectionBtn.disabled = true;
                stopCollectionBtn.disabled = false;
                // Start polling for status
                startCollectionStatusPolling();
            } else {
                alert('Failed to start collection: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // Stop collection button
    stopCollectionBtn.addEventListener('click', function() {
        fetch('/stop_collection', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                collectionStatus.style.display = 'none';
                startCollectionBtn.disabled = false;
                stopCollectionBtn.disabled = true;
                // Update collected people list
                loadCollectedPeople();
            } else {
                alert('Failed to stop collection: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // Train model button
    trainModelBtn.addEventListener('click', function() {
        const epochs = parseInt(epochsInput.value) || 20;
        const batchSize = parseInt(batchSizeInput.value) || 32;

        fetch('/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                epochs: epochs,
                batch_size: batchSize
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                trainingStatus.style.display = 'block';
                trainModelBtn.disabled = true;
                // Start polling for training status
                startTrainingStatusPolling();
            } else {
                alert('Failed to start training: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // Load collected people on page load
    loadCollectedPeople();

    // Load model info on page load
    loadModelInfo();

    // Load emotion model info on page load
    loadEmotionModelInfo();

    // Collection status polling
    let collectionStatusInterval = null;

    function startCollectionStatusPolling() {
        // Clear any existing interval
        if (collectionStatusInterval) {
            clearInterval(collectionStatusInterval);
        }

        // Poll every second
        collectionStatusInterval = setInterval(updateCollectionStatus, 1000);

        // Initial update
        updateCollectionStatus();
    }

    function updateCollectionStatus() {
        fetch('/collection_status')
        .then(response => response.json())
        .then(data => {
            if (data.active) {
                collectionStatusText.textContent = `Collecting: ${data.count}/${data.max} images for ${data.person}`;
                collectionProgress.style.width = `${data.progress}%`;

                // If collection is complete, stop polling and update UI
                if (data.count >= data.max) {
                    clearInterval(collectionStatusInterval);
                    collectionStatusInterval = null;

                    // Update UI
                    collectionStatus.style.display = 'none';
                    startCollectionBtn.disabled = false;
                    stopCollectionBtn.disabled = true;

                    // Update collected people list
                    loadCollectedPeople();
                }
            } else {
                // Collection not active, stop polling
                clearInterval(collectionStatusInterval);
                collectionStatusInterval = null;

                // Update UI
                collectionStatus.style.display = 'none';
                startCollectionBtn.disabled = false;
                stopCollectionBtn.disabled = true;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Training status polling
    let trainingStatusInterval = null;

    function startTrainingStatusPolling() {
        // Clear any existing interval
        if (trainingStatusInterval) {
            clearInterval(trainingStatusInterval);
        }

        // Poll every 2 seconds
        trainingStatusInterval = setInterval(updateTrainingStatus, 2000);

        // Initial update
        updateTrainingStatus();
    }

    function updateTrainingStatus() {
        fetch('/training_status')
        .then(response => response.json())
        .then(data => {
            if (data.is_training) {
                trainingStatusText.textContent = `Training: ${data.status} (${data.progress}%)`;
                trainingProgress.style.width = `${data.progress}%`;

                // Show the latest log entries if available
                if (data.log && data.log.length > 0) {
                    const latestLog = data.log[data.log.length - 1];
                    trainingStatusText.textContent += `\n${latestLog}`;
                }
            } else {
                // Training not active or complete
                if (data.status === "Training complete") {
                    trainingStatusText.textContent = "Training completed successfully!";
                    trainingProgress.style.width = "100%";

                    // Update model info
                    loadModelInfo();
                } else if (data.status.startsWith("Error")) {
                    trainingStatusText.textContent = `Training failed: ${data.status}`;
                }

                // Stop polling
                clearInterval(trainingStatusInterval);
                trainingStatusInterval = null;

                // Re-enable train button
                trainModelBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Load collected people
    function loadCollectedPeople() {
        fetch('/collected_people')
        .then(response => response.json())
        .then(people => {
            if (people.length === 0) {
                collectedPeople.innerHTML = '<p>No face data collected yet.</p>';
                return;
            }

            let html = '';
            people.forEach(person => {
                html += `
                    <div class="person-card">
                        <h4>${person.name}</h4>
                        <p>${person.image_count} images</p>
                    </div>
                `;
            });

            collectedPeople.innerHTML = html;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Load model info
    function loadModelInfo() {
        fetch('/model_info')
        .then(response => response.json())
        .then(info => {
            if (info.loaded) {
                modelInfoText.textContent = info.summary;
            } else {
                modelInfoText.textContent = "No model loaded";
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Load emotion model info
    function loadEmotionModelInfo() {
        fetch('/emotion_model_info')
        .then(response => response.json())
        .then(info => {
            if (info.loaded) {
                emotionModelInfoText.textContent = info.summary;
            } else {
                emotionModelInfoText.textContent = "No emotion model loaded";
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Train emotion model button
    trainEmotionBtn.addEventListener('click', function() {
        const epochs = parseInt(emotionEpochsInput.value) || 50;
        const batchSize = parseInt(emotionBatchSizeInput.value) || 32;

        fetch('/train_emotion_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                epochs: epochs,
                batch_size: batchSize
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                emotionTrainingStatus.style.display = 'block';
                trainEmotionBtn.disabled = true;
                // Start polling for training status
                startEmotionTrainingStatusPolling();
            } else {
                alert('Failed to start emotion training: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // Emotion training status polling
    let emotionTrainingStatusInterval = null;

    function startEmotionTrainingStatusPolling() {
        // Clear any existing interval
        if (emotionTrainingStatusInterval) {
            clearInterval(emotionTrainingStatusInterval);
        }

        // Poll every 2 seconds
        emotionTrainingStatusInterval = setInterval(updateEmotionTrainingStatus, 2000);

        // Initial update
        updateEmotionTrainingStatus();
    }

    function updateEmotionTrainingStatus() {
        fetch('/emotion_training_status')
        .then(response => response.json())
        .then(data => {
            if (data.is_training) {
                emotionTrainingStatusText.textContent = `Training: ${data.status} (${data.progress}%)`;
                emotionTrainingProgress.style.width = `${data.progress}%`;

                // Show the latest log entries if available
                if (data.log && data.log.length > 0) {
                    const latestLog = data.log[data.log.length - 1];
                    emotionTrainingStatusText.textContent += `\n${latestLog}`;
                }
            } else {
                // Training not active or complete
                if (data.status === "Training complete") {
                    emotionTrainingStatusText.textContent = "Emotion model training completed successfully!";
                    emotionTrainingProgress.style.width = "100%";

                    // Load the trained model
                    loadTrainedEmotionModel();

                    // Update model info
                    loadEmotionModelInfo();
                } else if (data.status.startsWith("Error")) {
                    emotionTrainingStatusText.textContent = `Training failed: ${data.status}`;
                }

                // Stop polling
                clearInterval(emotionTrainingStatusInterval);
                emotionTrainingStatusInterval = null;

                // Re-enable train button
                trainEmotionBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Load trained emotion model
    function loadTrainedEmotionModel() {
        fetch('/load_emotion_model', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Emotion model loaded successfully');
            } else {
                console.error('Failed to load emotion model');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    // Add capture to gallery
    function addCaptureToGallery(filename) {
        const captureItem = document.createElement('div');
        captureItem.className = 'capture-item';

        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();

        captureItem.innerHTML = `
            <img src="${filename}?t=${timestamp}" alt="Captured image">
            <div class="capture-actions">
                <button class="download-btn" data-src="${filename}">
                    <i class="fas fa-download"></i>
                </button>
                <button class="delete-btn" data-src="${filename}">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;

        // Add to gallery
        captureGallery.appendChild(captureItem);

        // Add event listeners for download and delete
        captureItem.querySelector('.download-btn').addEventListener('click', function() {
            const link = document.createElement('a');
            link.href = this.dataset.src;
            link.download = this.dataset.src.split('/').pop();
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });

        captureItem.querySelector('.delete-btn').addEventListener('click', function() {
            // In a real app, you would send a request to delete the file on the server
            // For now, we'll just remove it from the UI
            captureItem.remove();
        });
    }
});
