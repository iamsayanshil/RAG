const recordBtn = document.getElementById("record-btn");
const textBoxDiv = document.getElementById("text-box-div");

let micStream = null;

recordBtn.addEventListener("click", async function () {
  // Enable microphone
  if (!recordBtn.classList.contains("listening")) {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      recordBtn.classList.add("listening");

      console.log("Mic enabled");
      textBoxDiv.style.display = "none";
    } catch (error) {
      console.error("Microphone permission denied:", error);
    }
  }

  // Disable microphone
  else {
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }

    recordBtn.classList.remove("listening");

    console.log("Mic disabled");
    textBoxDiv.style.display = "block";
    textBoxDiv.textContent =
      "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tlorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tlorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
  }
});

// ==========================================
// 2. Three.js: Ocean
// ==========================================

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0b6839, 0.02); // Blends the water into your green background

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 2, 10); 
camera.lookAt(0, -2, 0);

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.domElement.id = 'three-canvas';
document.body.appendChild(renderer.domElement);

// --- Lighting for the water ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xfee101, 1.5); // Warm yellow light
directionalLight.position.set(10, 10, -10); 
scene.add(directionalLight);

// --- Element: Animated Sea ---
const waterGeometry = new THREE.PlaneGeometry(120, 50, 250, 150);
const waterMaterial = new THREE.MeshStandardMaterial({ 
    color: 0x1ca3ec, // Tropical sea blue
    flatShading: true, // Creates the low-poly look
    transparent: true,
    opacity: 1,
    roughness: 0.5,
    metalness: 0.3
});

const water = new THREE.Mesh(waterGeometry, waterMaterial);
water.rotation.x = -Math.PI / 2;
water.position.set(0, -30, -5); // Positioned at the bottom of the screen
scene.add(water);

// Store original Z positions for the wave animation
const waterPositions = waterGeometry.attributes.position;
const waterOriginalZ = [];
for (let i = 0; i < waterPositions.count; i++) {
    waterOriginalZ.push(waterPositions.getZ(i));
}

// --- Animation Loop ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const elapsedTime = clock.getElapsedTime();

    // Animate Water Vertices (Waves)
    for (let i = 0; i < waterPositions.count; i++) {
        const x = waterPositions.getX(i);
        const y = waterPositions.getY(i);
        
        // Gentle wave math
        const wave1 = Math.sin(x * 0.2 + elapsedTime * 1.2) * 0.4;
        const wave2 = Math.cos(y * 0.3 + elapsedTime * 0.9) * 0.4;
        
        waterPositions.setZ(i, waterOriginalZ[i] + wave1 + wave2);
    }
    waterPositions.needsUpdate = true;

    renderer.render(scene, camera);
}
animate();

// --- Handle Window Resizing ---
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});