function showLoader(){
document.getElementById("loader").style.display="block";
}

function imageSelected(input){
if(input.files.length>0){
let label=document.getElementById("uploadLabel");
label.innerHTML="✔ Image Uploaded";
label.style.background="#2ecc71";
}
}
