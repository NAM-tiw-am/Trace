const API_BASE = '/api';

// State
let cases = [];
let persons = [];
let videos = [];
let sightings = [];
let activeJobs = [];
let pollInterval = null;

// Helper to format any storage path into a web URL
function formatMediaUrl(path) {
    if (!path) return '';
    let normalized = path.replace(/\\/g, '/');
    const storageIdx = normalized.indexOf('storage/');
    if (storageIdx !== -1) {
        return '/' + normalized.substring(storageIdx);
    }
    return normalized.startsWith('/') ? normalized : '/' + normalized;
}

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadDashboard();
    startJobPolling();
});

// Tab Navigation
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));

    const section = document.getElementById(`${tabId}Section`);
    if (section) section.classList.add('active');

    const titles = {
        dashboard: ['Dashboard Overview', 'Real-time surveillance analytics and active facial search alerts'],
        cases: ['Cases Management', 'Organize and oversee missing person investigation cases'],
        persons: ['Missing Persons Registry', 'Manage registered subjects, reference portraits, and facial embeddings'],
        videos: ['CCTV Video Footage', 'Upload, index, and run AI search on surveillance streams'],
        sightings: ['Facial Matches & Sightings', 'Review detected faces, inspect evidence snapshots, and verify alerts'],
    };

    if (titles[tabId]) {
        document.getElementById('currentTabTitle').textContent = titles[tabId][0];
        document.getElementById('currentTabDesc').textContent = titles[tabId][1];
    }

    if (tabId === 'dashboard') loadDashboard();
    else if (tabId === 'cases') loadCases();
    else if (tabId === 'persons') loadPersons();
    else if (tabId === 'videos') loadVideos();
    else if (tabId === 'sightings') loadSightings();
}

// Notifications
function showToast(msg, isError = false) {
    const toast = document.getElementById('toastNotification');
    toast.textContent = msg;
    toast.style.borderColor = isError ? 'var(--danger)' : 'var(--primary)';
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3500);
}

// Modals
function openModal(id) {
    document.getElementById(id).classList.add('active');
    if (id === 'newPersonModal' || id === 'uploadVideoModal') {
        populateCaseDropdowns();
    }
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

function handleFileSelected(input, labelId) {
    if (input.files && input.files[0]) {
        document.getElementById(labelId).textContent = `Selected: ${input.files[0].name}`;
    }
}

// Populate Cases Dropdowns
async function populateCaseDropdowns() {
    try {
        const res = await fetch(`${API_BASE}/cases`);
        const data = await res.json();
        const personSelect = document.getElementById('personCaseId');
        const videoSelect = document.getElementById('videoCaseId');

        const options = data.map(c => `<option value="${c.id}">${c.case_number} — ${c.title || 'Untitled'}</option>`).join('');
        if (personSelect) personSelect.innerHTML = options || '<option value="">No cases created</option>';
        if (videoSelect) videoSelect.innerHTML = `<option value="">None (Independent feed)</option>` + options;
    } catch (e) {
        console.error(e);
    }
}

// API Loader: Dashboard
async function loadDashboard() {
    try {
        const [cRes, pRes, vRes, sRes] = await Promise.all([
            fetch(`${API_BASE}/cases`),
            fetch(`${API_BASE}/persons`),
            fetch(`${API_BASE}/videos`),
            fetch(`${API_BASE}/sightings`),
        ]);

        cases = await cRes.json();
        persons = await pRes.json();
        videos = await vRes.json();
        sightings = await sRes.json();

        document.getElementById('statCases').textContent = cases.length;
        document.getElementById('statPersons').textContent = persons.length;
        document.getElementById('statVideos').textContent = videos.length;
        document.getElementById('statSightings').textContent = sightings.length;
        document.getElementById('sightingsBadge').textContent = sightings.length;

        renderRecentSightings(sightings.slice(0, 6));
    } catch (e) {
        console.error('Failed to load dashboard:', e);
    }
}

// Render Recent Sightings
function renderRecentSightings(items) {
    const container = document.getElementById('recentSightingsGrid');
    if (!items || items.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-radar fa-2x"></i>
                <p>No sightings detected yet. Upload CCTV footage and run AI search.</p>
            </div>`;
        return;
    }

    container.innerHTML = items.map(s => renderSightingCardHTML(s)).join('');
}

function renderSightingCardHTML(s) {
    const personName = s.person ? s.person.name : `Person #${s.person_id}`;
    const cameraName = s.video ? s.video.camera_name : `Video #${s.video_id}`;
    const scorePct = (s.similarity_score * 100).toFixed(1);
    const snapUrl = formatMediaUrl(s.snapshot_path);

    return `
        <div class="sighting-card">
            <div class="sighting-image-wrap">
                ${snapUrl ? `<img src="${snapUrl}" alt="Sighting snapshot">` : `<i class="fa-solid fa-image-slash fa-2x text-muted"></i>`}
                <span class="similarity-badge"><i class="fa-solid fa-check"></i> ${scorePct}%</span>
                <span class="timestamp-badge"><i class="fa-solid fa-clock"></i> ${s.timestamp_formatted}</span>
            </div>
            <div class="sighting-content">
                <div class="flex-between">
                    <span class="sighting-person-name">${personName}</span>
                    <span class="status-tag ${s.match_status}">${s.match_status}</span>
                </div>
                <div class="sighting-meta">
                    <div><i class="fa-solid fa-video"></i> ${cameraName}</div>
                    <div><i class="fa-solid fa-film"></i> Frame ${s.frame_number} (${s.timestamp_seconds.toFixed(1)}s)</div>
                </div>
                <div class="sighting-actions">
                    <button class="btn btn-sm btn-outline" onclick="openSightingDetail(${s.id})">
                        <i class="fa-solid fa-magnifying-glass"></i> Inspect
                    </button>
                    <button class="btn btn-sm btn-success" onclick="updateSightingStatus(${s.id}, 'CONFIRMED')">
                        <i class="fa-solid fa-check"></i> Confirm
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="updateSightingStatus(${s.id}, 'REJECTED')">
                        <i class="fa-solid fa-xmark"></i> Reject
                    </button>
                </div>
            </div>
        </div>
    `;
}

// Cases Tab
async function loadCases() {
    try {
        const res = await fetch(`${API_BASE}/cases`);
        cases = await res.json();
        const tbody = document.getElementById('casesTableBody');

        if (cases.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No investigation cases created yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = cases.map(c => `
            <tr>
                <td><strong>${c.case_number}</strong></td>
                <td>${c.title || 'Untitled'}</td>
                <td><span class="status-tag ${c.status}">${c.status}</span></td>
                <td>${new Date(c.created_at).toLocaleDateString()}</td>
                <td>
                    <button class="btn btn-sm btn-outline" onclick="deleteCase(${c.id})"><i class="fa-solid fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    } catch (e) {
        showToast('Error loading cases', true);
    }
}

async function handleCreateCase(e) {
    e.preventDefault();
    const case_number = document.getElementById('caseNumber').value.trim();
    const title = document.getElementById('caseTitle').value.trim();
    const description = document.getElementById('caseDescription').value.trim();

    try {
        const res = await fetch(`${API_BASE}/cases`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ case_number, title, description }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to create case');
        }

        showToast('Case created successfully!');
        closeModal('newCaseModal');
        e.target.reset();
        loadCases();
        loadDashboard();
    } catch (err) {
        showToast(err.message, true);
    }
}

async function deleteCase(id) {
    if (!confirm('Are you sure you want to delete this case?')) return;
    try {
        await fetch(`${API_BASE}/cases/${id}`, { method: 'DELETE' });
        showToast('Case deleted.');
        loadCases();
        loadDashboard();
    } catch (e) {
        showToast('Error deleting case', true);
    }
}

// Persons Tab
async function loadPersons() {
    try {
        const res = await fetch(`${API_BASE}/persons`);
        persons = await res.json();
        const grid = document.getElementById('personsGrid');

        if (persons.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-user-slash fa-2x"></i>
                    <p>No persons registered. Register a missing person to begin face search.</p>
                </div>`;
            return;
        }

        grid.innerHTML = persons.map(p => {
            const photoUrl = p.photos && p.photos.length > 0 
                ? formatMediaUrl(p.photos[0].file_path) 
                : '';
            const embCount = p.embeddings ? p.embeddings.length : 0;

            return `
                <div class="person-card">
                    <div class="person-header">
                        ${photoUrl ? `<img src="${photoUrl}" class="person-avatar" alt="${p.name}">` : `<div class="person-avatar" style="display:flex;align-items:center;justify-content:center"><i class="fa-solid fa-user"></i></div>`}
                        <div class="person-info">
                            <h4>${p.name}</h4>
                            <span>Age: ${p.age || 'N/A'} • ${p.gender || 'Unknown'}</span>
                        </div>
                    </div>
                    <div class="person-meta">
                        <div><i class="fa-solid fa-location-dot"></i> Last seen: ${p.last_known_location || 'Not specified'}</div>
                        <div><i class="fa-solid fa-fingerprint"></i> Embeddings: <strong>${embCount} 512-D vector(s)</strong></div>
                    </div>
                    <div class="flex-between" style="margin-top:auto">
                        <button class="btn btn-sm btn-primary" onclick="openPhotoUploadModal(${p.id})">
                            <i class="fa-solid fa-camera"></i> Add Photo
                        </button>
                        <button class="btn btn-sm btn-outline" onclick="deletePerson(${p.id})">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        showToast('Error loading persons', true);
    }
}

async function handleCreatePerson(e) {
    e.preventDefault();
    const case_id = parseInt(document.getElementById('personCaseId').value);
    const name = document.getElementById('personName').value.trim();
    const age = parseInt(document.getElementById('personAge').value) || null;
    const gender = document.getElementById('personGender').value || null;
    const last_known_location = document.getElementById('personLocation').value.trim() || null;
    const description = document.getElementById('personDescription').value.trim() || null;

    try {
        const res = await fetch(`${API_BASE}/persons`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ case_id, name, age, gender, last_known_location, description }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to register person');
        }

        const newPerson = await res.json();
        showToast('Person registered! Now upload a reference photo.');
        closeModal('newPersonModal');
        e.target.reset();
        loadPersons();
        loadDashboard();

        // Immediately prompt for photo upload
        openPhotoUploadModal(newPerson.id);
    } catch (err) {
        showToast(err.message, true);
    }
}

function openPhotoUploadModal(personId) {
    document.getElementById('uploadPhotoPersonId').value = personId;
    document.getElementById('photoFileLabel').textContent = 'Click or drop portrait photo here';
    document.getElementById('personPhotoInput').value = '';
    openModal('uploadPhotoModal');
}

async function handleUploadPhoto(e) {
    e.preventDefault();
    const personId = document.getElementById('uploadPhotoPersonId').value;
    const fileInput = document.getElementById('personPhotoInput');
    const btn = document.getElementById('btnUploadPhoto');

    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select a photo file', true);
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Extracting ArcFace Embedding...';

    try {
        const res = await fetch(`${API_BASE}/persons/${personId}/photos`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to upload photo');
        }

        showToast('Face detected and 512-D embedding saved to PostgreSQL!');
        closeModal('uploadPhotoModal');
        loadPersons();
        loadDashboard();
    } catch (err) {
        showToast(err.message, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Extract & Store Embedding';
    }
}

async function deletePerson(id) {
    if (!confirm('Are you sure you want to delete this person and all their photos/sightings?')) return;
    try {
        await fetch(`${API_BASE}/persons/${id}`, { method: 'DELETE' });
        showToast('Person deleted.');
        loadPersons();
        loadDashboard();
    } catch (e) {
        showToast('Error deleting person', true);
    }
}

// Videos Tab
async function loadVideos() {
    try {
        const res = await fetch(`${API_BASE}/videos`);
        videos = await res.json();
        const grid = document.getElementById('videosGrid');

        if (videos.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-film fa-2x"></i>
                    <p>No CCTV videos uploaded yet. Upload MP4 videos to initiate facial scan.</p>
                </div>`;
            return;
        }

        grid.innerHTML = videos.map(v => `
            <div class="video-card">
                <div class="flex-between">
                    <h4>${v.camera_name}</h4>
                    <span class="status-tag ${v.status}">${v.status}</span>
                </div>
                <div class="person-meta">
                    <div><i class="fa-solid fa-location-dot"></i> ${v.location || 'Unknown location'}</div>
                    <div><i class="fa-solid fa-film"></i> ${v.total_frames} frames @ ${v.fps.toFixed(1)} FPS (${v.duration_seconds.toFixed(1)}s)</div>
                    <div><i class="fa-solid fa-calendar"></i> Uploaded: ${new Date(v.created_at).toLocaleDateString()}</div>
                </div>
                <div class="flex-between" style="margin-top:auto">
                    <button class="btn btn-sm btn-accent" onclick="startVideoProcessing(${v.id})">
                        <i class="fa-solid fa-play"></i> Run Face Search
                    </button>
                    <button class="btn btn-sm btn-outline" onclick="deleteVideo(${v.id})">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
    } catch (e) {
        showToast('Error loading videos', true);
    }
}

async function handleUploadVideo(e) {
    e.preventDefault();
    const camera_name = document.getElementById('videoCameraName').value.trim();
    const location = document.getElementById('videoLocation').value.trim();
    const case_id = document.getElementById('videoCaseId').value;
    const fileInput = document.getElementById('videoFileInput');
    const btn = document.getElementById('btnUploadVideo');

    if (!fileInput.files || !fileInput.files[0]) {
        showToast('Please select a video file', true);
        return;
    }

    const formData = new FormData();
    formData.append('camera_name', camera_name);
    if (location) formData.append('location', location);
    if (case_id) formData.append('case_id', case_id);
    formData.append('file', fileInput.files[0]);

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading Video...';

    try {
        const res = await fetch(`${API_BASE}/videos`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to upload video');
        }

        const newVideo = await res.json();
        showToast('Video uploaded and indexed successfully!');
        closeModal('uploadVideoModal');
        e.target.reset();
        loadVideos();
        loadDashboard();

        // Prompt to start processing immediately
        if (confirm('Video uploaded. Do you want to run AI facial detection on it now?')) {
            startVideoProcessing(newVideo.id);
        }
    } catch (err) {
        showToast(err.message, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Upload CCTV Video';
    }
}

async function startVideoProcessing(videoId) {
    try {
        const res = await fetch(`${API_BASE}/videos/${videoId}/process`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to start video processing');
        }

        showToast('AI Video processing queued and running in background!');
        loadVideos();
        pollJobs();
    } catch (err) {
        showToast(err.message, true);
    }
}

async function deleteVideo(id) {
    if (!confirm('Are you sure you want to delete this video?')) return;
    try {
        await fetch(`${API_BASE}/videos/${id}`, { method: 'DELETE' });
        showToast('Video deleted.');
        loadVideos();
        loadDashboard();
    } catch (e) {
        showToast('Error deleting video', true);
    }
}

// Job Polling
function startJobPolling() {
    pollJobs();
    pollInterval = setInterval(pollJobs, 2500);
}

async function pollJobs() {
    try {
        const res = await fetch(`${API_BASE}/jobs`);
        const jobs = await res.json();
        const active = jobs.filter(j => j.status === 'PROCESSING' || j.status === 'QUEUED');

        const card = document.getElementById('activeJobsCard');
        const list = document.getElementById('jobsList');

        if (active.length > 0) {
            card.style.display = 'block';
            list.innerHTML = active.map(j => `
                <div class="job-item">
                    <div class="job-header">
                        <strong>Job #${j.id} (Video #${j.video_id})</strong>
                        <span class="status-tag ${j.status}">${j.status}</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: ${j.progress}%"></div>
                    </div>
                    <div class="job-stats">
                        <span>Progress: <strong>${j.progress.toFixed(1)}%</strong></span>
                        <span>Frames: <strong>${j.current_frame} / ${j.total_frames}</strong></span>
                        <span>Faces: <strong>${j.faces_detected}</strong></span>
                        <span>Matches: <strong>${j.matches_found}</strong></span>
                    </div>
                </div>
            `).join('');
        } else {
            card.style.display = 'none';
        }
    } catch (e) {
        // Silent poll error
    }
}

// Sightings Tab
async function loadSightings() {
    try {
        const filter = document.getElementById('sightingStatusFilter').value;
        let url = `${API_BASE}/sightings`;
        if (filter) url += `?match_status=${filter}`;

        const res = await fetch(url);
        sightings = await res.json();
        const grid = document.getElementById('allSightingsGrid');

        if (sightings.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-eye fa-2x"></i>
                    <p>No sightings found matching criteria.</p>
                </div>`;
            return;
        }

        grid.innerHTML = sightings.map(s => renderSightingCardHTML(s)).join('');
    } catch (e) {
        showToast('Error loading sightings', true);
    }
}

async function updateSightingStatus(id, newStatus) {
    try {
        const res = await fetch(`${API_BASE}/sightings/${id}/status`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ match_status: newStatus }),
        });

        if (!res.ok) throw new Error('Failed to update status');

        showToast(`Sighting status updated to ${newStatus}`);
        loadSightings();
        loadDashboard();
    } catch (err) {
        showToast(err.message, true);
    }
}

// Sighting Detail Modal (Video & Snapshot Inspection)
async function openSightingDetail(id) {
    try {
        const res = await fetch(`${API_BASE}/sightings/${id}`);
        const s = await res.json();

        const title = document.getElementById('modalSightingTitle');
        const body = document.getElementById('modalSightingBody');

        title.textContent = `Sighting #${s.id} — ${s.person ? s.person.name : 'Unknown'}`;
        const snapUrl = formatMediaUrl(s.snapshot_path);
        const videoUrl = s.video && s.video.file_path ? formatMediaUrl(s.video.file_path) : '';

        body.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div>
                    <h4 style="margin-bottom: 10px;"><i class="fa-solid fa-camera"></i> Facial Detection Snapshot</h4>
                    <div style="background:#000; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; max-height: 300px;">
                        ${snapUrl ? `<img src="${snapUrl}" style="max-width:100%; max-height:300px; object-fit:contain;">` : '<p class="text-muted">No snapshot</p>'}
                    </div>
                </div>
                <div>
                    <h4 style="margin-bottom: 10px;"><i class="fa-solid fa-circle-info"></i> Sighting Metadata</h4>
                    <div style="background: rgba(255,255,255,0.03); padding: 16px; border-radius: 8px; font-size: 0.9rem; display: flex; flex-direction: column; gap: 8px;">
                        <div><strong>Person:</strong> ${s.person ? s.person.name : 'N/A'} (ID: ${s.person_id})</div>
                        <div><strong>Camera:</strong> ${s.video ? s.video.camera_name : 'N/A'}</div>
                        <div><strong>Timestamp:</strong> ${s.timestamp_formatted} (${s.timestamp_seconds.toFixed(2)}s)</div>
                        <div><strong>Frame:</strong> ${s.frame_number}</div>
                        <div><strong>Similarity Score:</strong> ${(s.similarity_score * 100).toFixed(2)}%</div>
                        <div><strong>Review Status:</strong> <span class="status-tag ${s.match_status}">${s.match_status}</span></div>
                        <div><strong>Notes:</strong> ${s.notes || 'None'}</div>
                    </div>
                </div>
            </div>

            ${videoUrl ? `
                <div style="margin-top: 20px;">
                    <h4 style="margin-bottom: 10px;"><i class="fa-solid fa-play"></i> Surveillance Video Playback (at timestamp ${s.timestamp_formatted})</h4>
                    <video id="detailVideoPlayer" controls style="width: 100%; border-radius: 8px; max-height: 350px; background: #000;">
                        <source src="${videoUrl}" type="video/mp4">
                        Your browser does not support HTML5 video.
                    </video>
                </div>
            ` : ''}

            <div class="modal-actions" style="margin-top: 24px;">
                <button class="btn btn-outline" onclick="closeModal('sightingDetailModal')">Close</button>
                <button class="btn btn-success" onclick="updateSightingStatus(${s.id}, 'CONFIRMED'); closeModal('sightingDetailModal');">Confirm Match</button>
                <button class="btn btn-danger" onclick="updateSightingStatus(${s.id}, 'REJECTED'); closeModal('sightingDetailModal');">Reject Match</button>
            </div>
        `;

        openModal('sightingDetailModal');

        // Automatically seek to sighting timestamp
        setTimeout(() => {
            const player = document.getElementById('detailVideoPlayer');
            if (player && s.timestamp_seconds) {
                player.currentTime = Math.max(0, s.timestamp_seconds - 1.0);
            }
        }, 300);

    } catch (e) {
        showToast('Error loading sighting details', true);
    }
}
