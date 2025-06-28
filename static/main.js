document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("imageInput");
  const previewWrapper = document.getElementById("previewWrapper");

  fileInput.addEventListener("change", () => {
    previewWrapper.innerHTML = "";
    const file = fileInput.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.createElement("img");
        img.src = e.target.result;
        img.classList.add("img-fluid", "shadow");
        previewWrapper.appendChild(img);
      };
      reader.readAsDataURL(file);
    }
  });
});
