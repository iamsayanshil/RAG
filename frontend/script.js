const recordBtn=document.getElementById('record-btn');

recordBtn.addEventListener('click',function(){
    recordBtn.classList.toggle('listening');
    console.log("clicked");
    
});