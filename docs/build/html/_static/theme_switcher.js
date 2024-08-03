// _static/theme_switcher.js

document.addEventListener("DOMContentLoaded", function () {
	const lightLogoPath = "_static/images/logo_light.svg";
	const darkLogoPath = "_static/images/logo_dark.svg";

	function getRelativePath() {
		const scripts = document.getElementsByTagName("script");
		for (let i = 0; i < scripts.length; i++) {
			if (scripts[i].src && scripts[i].src.includes("theme_switcher.js")) {
				const scriptPath = scripts[i].src;
				const relativePath = scriptPath.replace(/_static\/theme_switcher.js$/, "");
				return relativePath;
			}
		}
		return "";
	}

	const relativePath = getRelativePath();
	const lightLogo = relativePath + lightLogoPath;
	const darkLogo = relativePath + darkLogoPath;

	function updateLogo(theme) {
		const logo = document.querySelector(".logo img");
		if (logo) {
			if (theme === "dark") {
				logo.src = darkLogo;
			} else {
				logo.src = lightLogo;
			}
		}
	}

	// Detect initial theme and set logo
	const theme = document.documentElement.getAttribute("data-theme") || "light";
	updateLogo(theme);

	// Observe changes to the theme attribute
	const observer = new MutationObserver((mutations) => {
		mutations.forEach((mutation) => {
			if (mutation.attributeName === "data-theme") {
				const newTheme = document.documentElement.getAttribute("data-theme");
				updateLogo(newTheme);
			}
		});
	});

	observer.observe(document.documentElement, { attributes: true });
});
