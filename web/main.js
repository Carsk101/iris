// Version State Management
const savedVersion = localStorage.getItem('iris_version') || 'base';
if (savedVersion === 'salience') {
    // We add to documentElement in case body isn't parsed yet
    document.documentElement.classList.add('version-salience');
}

document.addEventListener('DOMContentLoaded', () => {
    // Ensure body respects the state
    if (savedVersion === 'salience') {
        document.body.classList.add('version-salience');
    }

    const currentLabels = document.querySelectorAll('.version-current');
    const optionBtns = document.querySelectorAll('.version-option');
    const navLefts = document.querySelectorAll('.nav-left');

    const updateUIs = (version) => {
        navLefts.forEach(nav => {
            nav.textContent = version === 'salience' ? 'iris + salience' : 'iris';
        });
        currentLabels.forEach(label => {
            label.textContent = version === 'salience' ? 'iris + salience' : 'iris';
        });
    };

    // Initial update
    updateUIs(savedVersion);

    optionBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Prevent default just in case, though it's a div
            e.preventDefault();
            const newVersion = e.currentTarget.getAttribute('data-value');
            localStorage.setItem('iris_version', newVersion);
            
            if (newVersion === 'salience') {
                document.documentElement.classList.add('version-salience');
                document.body.classList.add('version-salience');
            } else {
                document.documentElement.classList.remove('version-salience');
                document.body.classList.remove('version-salience');
            }
            
            updateUIs(newVersion);
            
            // Force hide the dropdown momentarily to reset hover state
            const parentDropdown = e.target.closest('.version-options');
            if (parentDropdown) {
                parentDropdown.style.display = 'none';
                setTimeout(() => { parentDropdown.style.display = ''; }, 100);
            }
        });
    });

    // Intersection Observer for scroll reveal animations
    const revealOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            } else {
                // Remove class to allow re-triggering when scrolling back
                entry.target.classList.remove('visible');
            }
        });
    }, revealOptions);

    const revealElements = document.querySelectorAll('.reveal');
    revealElements.forEach(el => revealObserver.observe(el));

    // Staggered reveals for multiple elements inside a container
    const staggerContainers = document.querySelectorAll('.stagger-reveal');
    staggerContainers.forEach(container => {
        const items = container.querySelectorAll('.reveal');
        items.forEach((item, index) => {
            item.style.transitionDelay = `${index * 150}ms`;
        });
    });

    // Hero and Cover reveal handling is now centralized in revealObserver
    const nav = document.querySelector('nav');

    // Nav visibility on scroll
    window.addEventListener('scroll', () => {
        if (window.scrollY > 200) {
            nav.classList.add('visible');
        } else {
            // Only hide on home page if we're at the top
            if (document.body.classList.contains('home')) {
                nav.classList.remove('visible');
            }
        }
    }, { passive: true });

    // Initial state: visible immediately on subpages, hidden on home
    if (!document.body.classList.contains('home')) {
        nav.classList.add('visible');
    } else {
        nav.classList.remove('visible');
    }

    // Hero headline stagger (for subpages)
    const heroHeadline = document.querySelector('.hero h1');
    const heroSubtitle = document.querySelector('.hero-subtitle');
    const heroExtras = document.querySelector('.hero-extras');

    if (heroHeadline && !document.body.classList.contains('home')) {
        setTimeout(() => heroHeadline.classList.add('visible'), 200);
    }
    if (heroSubtitle && !document.body.classList.contains('home')) {
        setTimeout(() => heroSubtitle.classList.add('visible'), 500);
    }
    if (heroExtras && !document.body.classList.contains('home')) {
        setTimeout(() => heroExtras.classList.add('visible'), 800);
    }

    // Custom "Gentle" Scroll Function
    const gentleScrollTo = (targetY, duration) => {
        const startY = window.scrollY;
        const difference = targetY - startY;
        let startTime = null;

        const animateScroll = (currentTime) => {
            if (!startTime) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const progress = Math.min(timeElapsed / duration, 1);
            
            // Easing: easeOutExpo (Starts fast, has a very long natural deceleration)
            const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);

            window.scrollTo(0, startY + difference * ease);

            if (timeElapsed < duration) {
                requestAnimationFrame(animateScroll);
            }
        };

        requestAnimationFrame(animateScroll);
    };

    // Console message
    console.log("%c iris v1.0 — perception first.", "color: #C4856A; font-style: italic; font-size: 14px;");
});
