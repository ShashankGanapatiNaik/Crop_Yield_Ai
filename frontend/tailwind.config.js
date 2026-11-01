const v = (name) => `rgb(var(--${name}) / <alpha-value>)`
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: v('bg'), surface: v('surface'), sunken: v('sunken'), ink: v('ink'), muted: v('muted'), line: v('line'),
        brand: v('brand'), 'brand-strong': v('brand-strong'), 'brand-soft': v('brand-soft'), 'on-brand': v('on-brand'),
        marigold: v('marigold'), danger: v('danger'), warn: v('warn'), info: v('info'), good: v('good'),
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', '"Noto Sans Devanagari"', '"Noto Sans Kannada"', 'system-ui', 'sans-serif'],
        sans: ['"Hanken Grotesk"', '"Noto Sans Devanagari"', '"Noto Sans Kannada"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
