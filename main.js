document.addEventListener('DOMContentLoaded', () => {
  // Intersection Observer for scroll animations
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.15
  };

  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        // Optional: Stop observing once it's visible if we only want it to animate once
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Observe all fade-in elements
  const fadeElements = document.querySelectorAll('.fade-in, .pipeline');
  fadeElements.forEach(el => {
    observer.observe(el);
  });

  // Code block copy functionality
  const preElements = document.querySelectorAll('pre');
  preElements.forEach(pre => {
    // Check if it already has a button (just in case)
    if (pre.querySelector('.copy-btn')) return;

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'copy';
    
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      if (code) {
        navigator.clipboard.writeText(code.innerText).then(() => {
          const originalText = btn.textContent;
          btn.textContent = 'copied';
          setTimeout(() => {
            btn.textContent = originalText;
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy code: ', err);
        });
      }
    });

    pre.appendChild(btn);
  });
});
