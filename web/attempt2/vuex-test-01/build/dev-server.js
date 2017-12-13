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
/*



@app.route('/pythonapi/search_catalog')
def hxr7s():
# &rows=2
    print("/pythonapi/search_catalog")
    q = request.args.get('q', type = str)#.replace('"', '\\"')
    n = request.args.get('n', default=15, type = int)
#     http://robinson.brontolabs.local:8983/solr/products/select?indent=on&q=%2BproductImageUrl:*%20productTitle:(white%20chair)
# n is multiplied here cos lots of products in database (from browse data) aren't in the map_productId_index.  why, idk.
    url = 'http://robinson.brontolabs.local:8983/solr/products/select?rows='+str(n*5)+'&q=%2BproductImageUrl:*%20%2BproductTitle:('+q+')'
    print("url")
    print(url)
    r = requests.get(url)
    data = r.json()
    docs = data['response']['docs']
#     print(data)
    outputDocs = []
    count = 1
    seenParentProductIds = set()
    for doc in docs:
        productId = doc['productId']
        productId = map_productId_parentProductId[productId]
        if productId in seenParentProductIds:
            continue
        seenParentProductIds.add(productId)
        if count > n:
            break
        if 'productImageUrl' in doc and productId in map_productId_index.keys():
            count += 1
            outputDocs.append({'productId': productId, 'productTitle': doc['productTitle'], 'productImageUrl': doc['productImageUrl'] if 'productImageUrl' in doc else ""})
    resp = flask.Response(str(json.dumps(outputDocs, indent=4)))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
#     return json.dumps(outputDocs, indent=4)  # sort_keys=True,

 */

// const url = 'http://robinson.brontolabs.local:8983/solr/products/select?rows=' + str(n * 5) + '&q=%2BproductImageUrl:*%20%2BproductTitle:(' + q + ')'

//stuart added
app.get('/api/search_catalog/:query/:n', function (req, res, next) {

  console.log('/api/search_catalog/:query/:n')
  const query = req.params.query
  const n = req.params.n

  console.log('query')
  console.log(query)
  console.log('n')
  console.log(n)

  console.log('/api/search_catalog/:query ' + query)
  const url = 'http://robinson.brontolabs.local:8983/solr/products/select?rows=' + (n * 5) + '&q=%2BproductImageUrl:*%20%2BproductTitle:(' + query + ')'

  console.log('url')
  console.log(url)

  console.log('getting url now!')
  axios.get(url)
    .then(response => {
      console.log('in response!:')
      console.log(response)
      console.log('response.data:')
      console.log(response.data)
      const data = response.data
      var seenParentProductIds = new Set()
      const docs = data['response']['docs']
      var outputDocs = []

      var count = 1
      for (var i = 0; i < docs.length; i++) {
        const doc = docs[i]
        productId = 'productParentProductId' in doc ? doc['productParentProductId'] : doc['productId']
        console.log('doc')
        console.log(doc)
        console.log('productId')
        console.log(productId)
        console.log('wtf')
        console.log(productId in seenParentProductIds)

        if (!(productId in seenParentProductIds)) {

          console.log('adding to ids set: ')
          console.log(seenParentProductIds)

          seenParentProductIds.add(productId)

          console.log('added to ids set: ')
          console.log(seenParentProductIds)
        }

        console.log('here1')
        if (count > n) {

          console.log('count > n')
          console.log(count)
          console.log(n)
          break
        }
        console.log('here2')

        if ('productImageUrl' in doc) {
          count++
          const outputDoc = {'productId': productId, 'productTitle': doc['productTitle'], 'productImageUrl': doc['productImageUrl']}

          console.log('outputDoc here')
          console.log(outputDoc)
          outputDocs.push(outputDoc)
          console.log('pushed here')

        }
        console.log('here3')

      }
      console.log('here4') // all this crap was cos set.push was killing me silently

      res.send(outputDocs)
    })
    .catch(error => res.send(error))
})

/*

@app.route('/pythonapi/recommendations/bronto')
def rhthr():
    print('/pythonapi/recommendations/bronto')
    print('request.args')
    print(request.args)
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split("|")
    productIdsInput = list(filter(None, productIdsInput))
#     [x+1 if x >= 45 else x+5 for x in l]
    productIdsInput = [map_productId_parentProductId[x] if x in map_productId_parentProductId else x for x in productIdsInput]
    numProductsToRecommend = request.args.get('n', default = 2, type = int)
    doReturnDetails = request.args.get('return_details', default = "true", type = str)

    recProductIds = getRecsBronto(productIdsInput, n=numProductsToRecommend, search_engine="solr")
    print("len(recProductIds)")
    print(len(recProductIds))
#     return json.dumps({'productIds': recProductIds})

    if doReturnDetails == "true":
        output = getProductsDetails(recProductIds)
        print("len(output)")
        print(len(output))
    else:
        output = json.dumps({'productIds': recProductIds})

    resp = flask.Response(output)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['lalallala'] = 'hieeeee'
    return resp


 */

// var fs = require('fs')
//
// var productIdsText = fs.readFileSync('./src/assets/productIds', 'utf-8')
// var productIdsTextByLine = productIdsText.split('\n')
//
// var parentProductIdsText = fs.readFileSync('./src/assets/parentProductIds', 'utf-8')
// var parentProductIdsTextByLine = parentProductIdsText.split('\n')
//
//
// var titlesText = fs.readFileSync('./src/assets/titles', 'utf-8')
// var titlesTextByLine = titlesText.split('\n')
//
// var imageUrlsText = fs.readFileSync('./src/assets/imageUrls', 'utf-8')
// var imageUrlsTextByLine = imageUrlsText.split('\n')
//
// // console.log(productIdsTextByLine)
// // console.log(parentProductIdsTextByLine)
//
// var map_productId_parentProductId = {}
// var map_parentProductId_productId = {}
// var map_productId_title = {}
// var map_productId_imageUrl = {}
//
//
// console.log('productIdsTextByLine.length')
// console.log(productIdsTextByLine.length)
//
// console.log('parentProductIdsTextByLine.length')
// console.log(parentProductIdsTextByLine.length)
//
// console.log('titlesTextByLine.length')
// console.log(titlesTextByLine.length)
//
// console.log('imageUrlsTextByLine.length')
// console.log(imageUrlsTextByLine.length)
//
//
// for (var i = 0; i < productIdsTextByLine.length; i++) {
//   if (parentProductIdsTextByLine[i].trim().length > 0) {
//     // console.log(i)
//     map_productId_parentProductId[productIdsTextByLine[i]] = parentProductIdsTextByLine[i].trim()
//     map_parentProductId_productId[parentProductIdsTextByLine[i].trim()] = productIdsTextByLine[i]
//   }
//   map_productId_title[productIdsTextByLine[i]] = titlesTextByLine[i]
//   if (imageUrlsTextByLine[i].trim().length > 0)
//     map_productId_imageUrl[productIdsTextByLine[i]] = imageUrlsTextByLine[i]
// }

// console.log('map_productId_parentProductId')
// console.log(map_productId_parentProductId)

function getRecBronto (productIdsInput) {
  var productIds = [...productIdsInput]

  var qEquals = 'vv_indicators:('
  for (var i = 0; i < productIds.length; i++) {
    var id = productIds[i]
    qEquals += '"' + (id) + '" '
  }
  qEquals += ')'
  const url = 'http://robinson.brontolabs.local:8983/solr/products/select?rows=' + (productIds.length + 5) + '&q=%2BproductImageUrl:*%20' + qEquals
  console.log('url')
  console.log(url)
  axios.get(url)
    .then(response => {
      const r = response.data

      var i = 0
      if (r['response']['numFound'] == 0) {
        return ''
      }

      var recProductId = r['response']['docs'][i]['productId']

      var productIdToCheck = recProductId

      /*

        while productIdToCheck in productIds and i < len(r["response"]["docs"]) - 1:
            i += 1
            recProductId = r["response"]["docs"][i]["productId"]
            if siteId=='35569':
                print("recProductId")
                print(recProductId)
                if recProductId not in map_productId_parentProductId:
                    continue
                productIdToCheck = map_productId_parentProductId[recProductId]
            else:
                productIdToCheck = recProductId
       */

      while (productIdToCheck in productIdsInput && i < r['response']['docs'].length - 1) {
        i++
        recProductId = r['response']['docs'][i]['productId']
        productIdToCheck = recProductId
      }

      return productIdToCheck
    })
    .catch(error => res.send(error))

}

function getBrontoRecs (productIdsInput, n) {
  var generated = [...productIdsInput]

  for (var i = 0; i < n; i++) {
    var rec = getRecBronto(generated)
    generated.push(rec)
  }
  return generated
}

function getProductsDetails (productIdsInput) {

  /*
   productDocs = []
      for productId in productIds:
          doc = {}
          doc['productId'] = productId
          doc['productImageUrl'] = map_productId_imgUrl[productId] if productId in map_productId_imgUrl else 'no url'
          doc['productTitle'] = map_productId_title[productId] if productId in map_productId_title else 'no title'
          if productId in map_productId_imgUrl:
              productDocs.append(doc)
      return json.dumps(productDocs, indent=4)  # sort_keys=True,
   */

  var productDocs = []

  for (var i = 0; i < productIdsInput.length; i++) {
    productId = productIdsInput[i]
    var doc = {}
    doc['productId'] = productId
    var imageUrl = ''
    var title = ''
    //is parent
    if (productId in map_parentProductId_productId) {
      var child = map_parentProductId_productId[productId]
      imageUrl = map_productId_imageUrl[child]
      title = map_productId_title[child]
    }
    else {
      imageUrl = map_productId_imageUrl[productId]
      title = map_productId_title[productId]

    }
    if (imageUrl == undefined || imageUrl == null || imageUrl.trim().length == 0) {
      continue
    }

    doc['productImageUrl'] = imageUrl
    doc['productTitle'] = title
    productDocs.push(doc)
  }
  return productDocs

}

app.get('/api/recommendations/bronto/:n/:productIdsInput', function (req, res, next) {

  console.log('/api/search_catalog/:query/:n')
  var productIdsInput0 = req.params.productIdsInput
  var n = req.params.n

  productIdsInput0 = productIdsInput0.split('|')

  var productIdsInput = []

  for (var i = 0; i < productIdsInput0.length; i++) {
    productId = productIdsInput0[i]
    if (productId.trim().length > 0) {

      if (productId in map_productId_parentProductId) {
        productId = map_productId_parentProductId[map_productId_parentProductId]
      }

      productIdsInput.push(productId)
    }
  }

  const recProductIds = getBrontoRecs(productIdsInput, n)

  var output = getProductsDetails(recProductIds)
})

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
