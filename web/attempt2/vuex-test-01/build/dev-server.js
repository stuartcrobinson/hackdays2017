require('./check-versions')()

var config = require('../config')

const axios = require('axios')

if (!process.env.NODE_ENV) {
  process.env.NODE_ENV = JSON.parse(config.dev.env.NODE_ENV)
}

var opn = require('opn')
var path = require('path')
var express = require('express')
var webpack = require('webpack')
var proxyMiddleware = require('http-proxy-middleware')
var webpackConfig = require('./webpack.dev.conf')

// default port where dev server listens for incoming traffic
var port = process.env.PORT || config.dev.port
// automatically open browser, if not set will be false
var autoOpenBrowser = !!config.dev.autoOpenBrowser
// Define HTTP proxies to your custom API backend
// https://github.com/chimurai/http-proxy-middleware
var proxyTable = config.dev.proxyTable

var app = express()

// axios.get(`/api/elasticsearchget/products/_search?q=${query}`) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`

// //stuart added
// app.get('/api/elasticsearchget/:path', function (req, res, next) {
//   axios.get(`http://robinson.brontolabs.local:9115/${req.params.path}`)
//     .then(response => res.send(response))
//     .catch(error => res.send(error))
//   // Handle the get for this route
//   res.send('hello')
//   next()
// })

/*
apis to add:



@app.route('/api/suggest')
    q = request.args.get('q', type = str)

@app.route('/api/search_catalog')
    q = request.args.get('q', type = str)#.replace('"', '\\"')
    n = request.args.get('n', default=15, type = int)

@app.route('/api/recommendations/ff')
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")
    numProductsToRecommend = request.args.get('n', default = 1, type = int)
    doReturnDetails = request.args.get('return_details', default = False, type = str)

@app.route('/api/recommendations/bronto')
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")
    numProductsToRecommend = request.args.get('n', default = 1, type = int)
    search_engine = request.args.get('search_engine', default = 'solr', type = str)
    doReturnDetails = request.args.get('return_details', default = False, type = str)

 */

var dispay = (response) => {
  console.log('response:')
  console.log(response)
  console.log('response.data:')
  console.log(response.data)
}

// const server = 'http://localhost:5000'
//
// app.get('/api/suggest/:query', function (req, res, next) {
//   console.log(`hit /api/suggest/${req.params.query}`)
//
//   axios.get(`${server}/pythonapi/suggest?q=${req.params.query}`)
//     .then(response => {
//       dispay(response)
//       res.send(response.data)
//     }).catch(error => res.send(error))
// })
//
// app.get('/api/search_catalog/:query', function (req, res, next) {
//   console.log(`/api/search_catalog/${req.params.query}`)
//   axios.get(`${server}/pythonapi/search_catalog?q=${req.params.query}`)
//     .then(response => {
//       dispay(response)
//       res.send(response.data)
//     }).catch(error => res.send(error))
// })
//
// app.get('/api/recommendations/ff/:query', function (req, res, next) {
//   console.log(`/api/search_catalog/${req.params.query}`)
//   axios.get(`${server}/pythonapi/search_catalog?q=${req.params.query}`)
//     .then(response => {
//       dispay(response)
//       res.send(response.data)
//     }).catch(error => res.send(error))
// })

/////////////////////////
//old - deprecated

//stuart added
app.get('/api/queryproducts/:query', function (req, res, next) {
  console.log('app.get(\'/api/queryproducts/:query\', function (req, res, next) {.  query: ' + req.params.query)
  axios.get(`http://robinson.brontolabs.local:9115/products/_search?q=${req.params.query}`)
    .then(response => {
      console.log('response:')
      console.log(response)
      console.log('response.data:')
      console.log(response.data)
      res.send(response.data)
    })
    .catch(error => res.send(error))
})

// used - this is faster than going through python flask
app.get('/api/queryproducttitletypeaheads/:query', function (req, res, next) {

  console.log('app.get(\'/api/queryproducttitletypeaheads/:query\', function (req, res, next) {.  query: ' + req.params.query)

  axios.post(`http://robinson.brontolabs.local:9115/products/_search?pretty`, {
    'suggest': {
      'mysuggest': {
        'prefix': req.params.query,
        'completion': {
          'field': 'titlesuggest'
        }
      }
    }
  }).then(response => {
    //create list of product titles only
    console.log('response:')
    console.log(response)

    const titles = response.data.suggest.mysuggest[0].options.map(el => el._source.titlesuggest)
    console.log('titles:')
    console.log(titles)
    res.send(titles)
  })
    .catch(error => res.send(error))
})

//queryproducttitletypeaheads

app.post('/', function (req, res, next) {
  // Handle the post for this route
})

// app.use(function (req, res, next) {
//   res.setHeader('Access-Control-Allow-Origin', '*')
//   res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE')
//   res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type, Authorization')
//   res.setHeader('Access-Control-Allow-Credentials', true)
//   if ('OPTIONS' === req.method) { res.send(204) } else { next() }
// })

var compiler = webpack(webpackConfig)

var devMiddleware = require('webpack-dev-middleware')(compiler, {
  publicPath: webpackConfig.output.publicPath,
  quiet: true
})

var hotMiddleware = require('webpack-hot-middleware')(compiler, {
  log: false,
  heartbeat: 2000
})
// force page reload when html-webpack-plugin template changes
compiler.plugin('compilation', function (compilation) {
  compilation.plugin('html-webpack-plugin-after-emit', function (data, cb) {
    hotMiddleware.publish({action: 'reload'})
    cb()
  })
})

// proxy api requests
Object.keys(proxyTable).forEach(function (context) {
  var options = proxyTable[context]
  if (typeof options === 'string') {
    options = {target: options}
  }
  app.use(proxyMiddleware(options.filter || context, options))
})

// handle fallback for HTML5 history API
app.use(require('connect-history-api-fallback')())

// serve webpack bundle output
app.use(devMiddleware)

// enable hot-reload and state-preserving
// compilation error display
app.use(hotMiddleware)

// serve pure static assets
var staticPath = path.posix.join(config.dev.assetsPublicPath, config.dev.assetsSubDirectory)
app.use(staticPath, express.static('./static'))

var uri = 'http://localhost:' + port

var _resolve
var readyPromise = new Promise(resolve => {
  _resolve = resolve
})

console.log('> Starting dev server...')
devMiddleware.waitUntilValid(() => {
  console.log('> Listening at ' + uri + '\n')
  // when env is testing, don't need open it
  if (autoOpenBrowser && process.env.NODE_ENV !== 'testing') {
    opn(uri)
  }
  _resolve()
})

// app.use(function(req, res, next) {
//   res.header("Access-Control-Allow-Origin", "*");
//   res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
//   next();
// });

var server = app.listen(port)

module.exports = {
  ready: readyPromise,
  close: () => {
    server.close()
  }
}
