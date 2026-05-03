/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f2f8ff",
          100: "#e6f0ff",
          200: "#bed8ff",
          300: "#8ebaff",
          400: "#5f96ff",
          500: "#3a75fb",
          600: "#295be0",
          700: "#2148b5",
          800: "#1e3f8f",
          900: "#1f3871"
        }
      },
      boxShadow: {
        soft: "0 10px 40px -12px rgba(59,130,246,0.35)"
      }
    }
  },
  plugins: []
};
