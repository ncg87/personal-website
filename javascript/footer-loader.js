// Used to load footer on the bottom of each page
document.addEventListener('DOMContentLoaded', function() {
    fetch('../html/footer.html')
      .then(response => response.text())
      .then(data => {
        document.body.insertAdjacentHTML('beforeend', data);
      })
      .catch(error => console.error('Error loading footer:', error));
  });