module.exports = {
  title: 'Foolbox',
  description: 'A Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, and JAX',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'API', link: 'https://foolbox.readthedocs.io/en/stable/' },
      { text: 'GitHub', link: 'https://github.com/bethgelab/foolbox' },
    ],
    sidebar: [
      {
        title: 'Guide',
        collapsable: false,
        children: [
          '/guide/',
          '/guide/getting-started',
          '/guide/examples',
          '/guide/development',
          '/guide/adding_attacks',
        ],
      },
    ],
  },
}
