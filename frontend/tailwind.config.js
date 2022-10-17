module.exports = {
  purge: ['./src/**/*.{js,jsx,ts,tsx}', './public/index.html'],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      colors: {
        'customPurp': '#7353BA',
        'customGreen': '#59A96A'
      }
    }
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
