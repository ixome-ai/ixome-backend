module.exports = ({ env }) => ({
  host: '0.0.0.0', // Bind to all interfaces
  port: env.int('PORT', 1337),
  app: {
    keys: env.array('APP_KEYS'),
  },
  admin: {
    url: '/admin',
  },
});