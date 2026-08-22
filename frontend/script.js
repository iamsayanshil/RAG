const recordBtn = document.getElementById("record-btn");

let micStream = null;

recordBtn.addEventListener("click", async function () {

    // Enable microphone
    if (!recordBtn.classList.contains("listening")) {

        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: true
            });

            recordBtn.classList.add("listening");

            console.log("🎤 Mic enabled");

        } catch (error) {
            console.error("Microphone permission denied:", error);
        }

    } 
    
    // Disable microphone
    else {

        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }

        recordBtn.classList.remove("listening");

        console.log("🔇 Mic disabled");
    }
});