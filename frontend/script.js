const recordBtn = document.getElementById("record-btn");
const textBoxDiv = document.getElementById("text-box-div");
const inputEl = document.getElementById("input");

/** @type {MediaStream | null} */
let micStream = null;
/** @type {AudioContext | null} */
let audioContext = null;
/** @type {ScriptProcessorNode | null} */
let scriptNode = null;
/** @type {Float32Array[]} */
let pcmBuffers = [];

const BACKEND_URL = "http://localhost:8000/api/v1/voice-query";

recordBtn.addEventListener("click", async function () {
  // Enable microphone & record PCM WAV
  if (!recordBtn.classList.contains("listening")) {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(micStream);

      scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
      pcmBuffers = [];

      scriptNode.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        pcmBuffers.push(new Float32Array(inputData));
      };

      source.connect(scriptNode);
      scriptNode.connect(audioContext.destination);

      recordBtn.classList.add("listening");

      console.log("Mic enabled & recording PCM WAV");
      if (inputEl) inputEl.value = "Recording speech... Speak your query!";
      if (textBoxDiv) textBoxDiv.style.display = "none";
    } catch (error) {
      console.error("Microphone permission denied:", error);
    }
  }

  // Disable microphone & process audio
  else {
    recordBtn.classList.remove("listening");
    if (inputEl) inputEl.value = "Processing speech with backend...";
    console.log("Mic disabled, processing WAV...");

    if (scriptNode) {
      scriptNode.disconnect();
      scriptNode = null;
    }

    const sampleRate = audioContext ? audioContext.sampleRate : 44100;

    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }

    if (audioContext) {
      await audioContext.close();
      audioContext = null;
    }

    // Flatten PCM buffers
    let totalSamples = 0;
    for (const buf of pcmBuffers) totalSamples += buf.length;
    const flattened = new Float32Array(totalSamples);
    let offset = 0;
    for (const buf of pcmBuffers) {
      flattened.set(buf, offset);
      offset += buf.length;
    }

    const wavBlob = encodeWav(flattened, sampleRate);
    await sendAudioToBackend(wavBlob);
  }
});

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let index = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    index += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

async function sendAudioToBackend(blob) {
  const formData = new FormData();
  formData.append("audio", blob, "recording.wav");

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to process audio.");
    }

    const data = await response.json();
    console.log("Backend Response:", data);

    if (inputEl) inputEl.value = data.transcript || data.query || "";
    if (textBoxDiv) {
      textBoxDiv.style.display = "block";
      textBoxDiv.textContent = data.answer || "No answer generated.";
    }
  } catch (error) {
    console.error("Error connecting to backend:", error);
    if (inputEl) inputEl.value = "Error connecting to backend.";
    if (textBoxDiv) {
      textBoxDiv.style.display = "block";
      textBoxDiv.textContent = error.message;
    }
  }
}

// ==========================================
// 2. Three.js: Ocean
// ==========================================

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0b6839, 0.02);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 2, 10); 
camera.lookAt(0, -2, 0);

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.domElement.id = 'three-canvas';
document.body.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xfee101, 1.5);
directionalLight.position.set(10, 10, -10); 
scene.add(directionalLight);

const waterGeometry = new THREE.PlaneGeometry(120, 50, 250, 150);
const waterMaterial = new THREE.MeshStandardMaterial({ 
    color: 0x1ca3ec,
    flatShading: true,
    transparent: true,
    opacity: 1,
    roughness: 0.5,
    metalness: 0.3
});

const water = new THREE.Mesh(waterGeometry, waterMaterial);
water.rotation.x = -Math.PI / 2;
water.position.set(0, -30, -5);
scene.add(water);

const waterPositions = waterGeometry.attributes.position;
const waterOriginalZ = [];
for (let i = 0; i < waterPositions.count; i++) {
    waterOriginalZ.push(waterPositions.getZ(i));
}

const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const elapsedTime = clock.getElapsedTime();

    for (let i = 0; i < waterPositions.count; i++) {
        const x = waterPositions.getX(i);
        const y = waterPositions.getY(i);
        
        const wave1 = Math.sin(x * 0.2 + elapsedTime * 1.2) * 0.4;
        const wave2 = Math.cos(y * 0.3 + elapsedTime * 0.9) * 0.4;
        
        waterPositions.setZ(i, waterOriginalZ[i] + wave1 + wave2);
    }
    waterPositions.needsUpdate = true;

    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});