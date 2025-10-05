window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

	var previewOptions = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
			breakpoints: [
				{
					changePoint: 1024,
					slidesToShow: 2,
				},
				{
					changePoint: 768,
					slidesToShow: 1,
				}
			]
		}

	var defaultOptions = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
		}

	const attachedReels = [];
	const previewInstances = bulmaCarousel.attach('#results-carousel.preview-carousel', previewOptions);
	const regularInstances = bulmaCarousel.attach('.carousel:not(.preview-carousel)', defaultOptions);
	previewInstances.forEach((carousel) => attachedReels.push(carousel));
	regularInstances.forEach((carousel) => attachedReels.push(carousel));

	bulmaSlider.attach();

	const selectorGroups = document.querySelectorAll('[data-carousel-select]');
	if (selectorGroups.length) {
		const carouselById = attachedReels.reduce((acc, carousel) => {
			if (carousel.element && carousel.element.id) {
				acc[carousel.element.id] = carousel;
			}
			return acc;
		}, {});

		selectorGroups.forEach((group) => {
			const carouselId = group.dataset.carouselSelect;
			const carousel = carouselById[carouselId];
			if (!carousel) {
				return;
			}

			const buttons = group.querySelectorAll('[data-slide]');
			const updateActiveButton = (activeIndex) => {
				buttons.forEach((button) => {
					const isActive = parseInt(button.dataset.slide, 10) === activeIndex;
					button.classList.remove('is-primary', 'is-light');
					button.classList.add(isActive ? 'is-primary' : 'is-light');
				});
			};

			const handleSync = () => {
				const state = carousel.state;
				const rawIndex = Number.isFinite(state.next) ? state.next : state.index;
				const activeIndex = ((rawIndex % state.length) + state.length) % state.length;
				updateActiveButton(activeIndex);
			};

			buttons.forEach((button) => {
				button.addEventListener('click', () => {
					const slideIndex = parseInt(button.dataset.slide, 10);
					if (Number.isFinite(slideIndex)) {
						carousel.state.next = slideIndex;
						carousel.show();
					}
				});
			});

			carousel.on('show', handleSync);
			carousel.on('after:show', handleSync);
			carousel.on('before:show', handleSync);

			handleSync();
		});
	}

	const lazyVideos = document.querySelectorAll('video[data-src]');
	const maybePlay = (video) => {
		if (video.dataset.autoplay === 'true') {
			const playPromise = video.play();
			if (playPromise && typeof playPromise.catch === 'function') {
				playPromise.catch(() => {});
			}
		}
	};

	if ('IntersectionObserver' in window && lazyVideos.length) {
		const observer = new IntersectionObserver((entries) => {
			entries.forEach((entry) => {
				if (entry.isIntersecting) {
					const video = entry.target;
					if (!video.dataset.loaded) {
						const source = video.querySelector('source');
						if (source) {
							source.src = video.dataset.src;
							video.load();
							video.dataset.loaded = 'true';
							if (video.dataset.loop === 'true') {
								video.loop = true;
							}
							maybePlay(video);
						}
					}
					observer.unobserve(video);
				}
			});
		}, {
			root: null,
			rootMargin: '200px',
			threshold: 0.1
		});

		lazyVideos.forEach((video) => observer.observe(video));
	} else {
		lazyVideos.forEach((video) => {
			const source = video.querySelector('source');
			if (source && !video.dataset.loaded) {
				source.src = video.dataset.src;
				video.load();
				video.dataset.loaded = 'true';
				if (video.dataset.loop === 'true') {
					video.loop = true;
				}
				maybePlay(video);
			}
		});
	}

})