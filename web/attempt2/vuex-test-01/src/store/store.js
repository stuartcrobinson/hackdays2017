import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'

Vue.use(Vuex)

// const server = 'http://localhost:5000'
const server = 'http://52.90.195.143:5000'

// const dodisplay = (response) => {
//   console.log('response:')
//   console.log(response)
//   console.log('response.data:')
//   console.log(response.data)
// }

const state = {
  count: 0,
  name: 'stuart',
  gifurl: '',
  isyes: false,
  isLoadingProductsSearch: [],
  isLoadingBrontoRecommendedProducts: [],
  isLoadingStuartRecommendedProducts: [],
  queriedProducts: [],
  browsedProducts: [],
  currentProduct: {},
  brontoRecommendedProducts: [],
  stuartRecommendedProducts: []
}

const getters = {
  evenOrOdd: state => state.count % 2 === 0 ? 'even' : 'odd'
}

// synchronous
const mutations = {
  increment (state) {
    state.count++
  },
  decrement (state) {
    state.count--
  },
  setgifurl (state, url) {
    state.gifurl = url
  },
  setisyes (state, isyes) {
    state.isyes = isyes
  },
  setIsLoadingProductsSearch (state, booleanvalue) {
    // state.isLoadingProductsSearch = booleanvalue
    if (booleanvalue) {
      state.isLoadingProductsSearch.push(true)
    } else {
      state.isLoadingProductsSearch.pop()
    }
  },
  setIsLoadingBrontoRecommendedProducts (state, booleanvalue) {
    if (booleanvalue) {
      state.isLoadingBrontoRecommendedProducts.push(true)
    } else {
      state.isLoadingBrontoRecommendedProducts.pop()
    }
  },
  setIsLoadingStuartRecommendedProducts (state, booleanvalue) {
    // state.isLoadingStuartRecommendedProducts = booleanvalue
    if (booleanvalue) {
      state.isLoadingStuartRecommendedProducts.push(true)
    } else {
      state.isLoadingStuartRecommendedProducts.pop()
    }
  },
  setQueriedProducts (state, queriedProducts) {
    state.queriedProducts = queriedProducts
  },
  setBrontoRecommendedProducts (state, products) {
    state.brontoRecommendedProducts = products
  },
  setStuartRecommendedProducts (state, products) {
    state.stuartRecommendedProducts = products
  },
  prependProductToBrowsedProducts (state, newProduct) {
    console.log('prepending!')
    if (newProduct['productId'] === undefined) {
      console.log('productId undefined in newProduct')
      return
    }
    if (newProduct['productId'] === null) {
      console.log('productId null in newProduct')
      return
    }
    if (newProduct['productId'].length === 0) {
      console.log('productId length === 0 in newProduct')
      return
    }
    if (state.browsedProducts.length === 0) {
      state.browsedProducts = [newProduct]
    } else {
      state.browsedProducts = [newProduct, ...state.browsedProducts]
    }
    console.log('browsed:')
    console.log(state.browsedProducts)
  },
  removeProductFromBrowsedProducts (state, product) {
    console.log('removing!')
    if (state.browsedProducts.length > 0) {
      state.browsedProducts = state.browsedProducts.filter(p => p['productId'] !== product['productId'])
    }
    console.log('browsed:')
    console.log(state.browsedProducts)

    if (state.currentProduct.productId === product.productId) {
      state.currentProduct = {}
    }
    // state.brontoRecommendedProducts = [] // doens't help
    // state.stuartRecommendedProducts = []
  },
  setCurrentProduct (state, product) {
    state.currentProduct = product
  }
}

// asynchronous - good for api stuff
const actions = {
  increment: ({commit}) => commit('increment'),
  decrement: ({commit}) => commit('decrement'),
  incrementIfOdd ({commit, state}) {
    if ((state.count + 1) % 2 === 0) {
      commit('increment')
    }
  },
  incrementAsync ({commit}) {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        commit('increment')
        resolve()
      }, 1000)
    })
  },
  yesnoaction (context, customer) {
    return new Promise((resolve, reject) => {
      axios.get('https://yesno.wtf/api')
        .then(function (res) {
          if (res.data.answer === 'yes') {
            context.commit('setisyes', true)
          } else {
            context.commit('setisyes', false)
          }
          context.commit('setgifurl', res.data.image)
        })
        .catch(function (err) {
          console.log(err)
        })
    })
  },
  search_catalog (context, query) {
    console.log('search_catalog')
    context.commit('setIsLoadingProductsSearch', true)
    context.commit('setQueriedProducts', {})
    return new Promise((resolve, reject) => {
      axios.get(`${server}/pythonapi/search_catalog?n=8&q=${query}`) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`
        .then(function (res) {
          context.commit('setIsLoadingProductsSearch', false)
          // dodisplay(res)
          context.commit('setQueriedProducts', res.data)
        })
        .catch(function (err) {
          console.log(err)
        })
    })
  },
  search_catalogExpress (context, query) {
    console.log('search_catalog')
    context.commit('setIsLoadingProductsSearch', true)
    context.commit('setQueriedProducts', {})
    return new Promise((resolve, reject) => {
      axios.get(`/api/search_catalog/${query}/${20}`) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`
        .then(function (res) {
          context.commit('setIsLoadingProductsSearch', false)
          // dodisplay(res)
          context.commit('setQueriedProducts', res.data)
        })
        .catch(function (err) {
          console.log(err)
        })
    })
  },
  getbrowsefunction (context, product) {
    console.log('hello!!!!!')
    const x = () => {
      this.browse(context, product)
      console.log('hi')
    }
    console.log(x)
    return x
  },
  removeProduct (context, product) {
    context.commit('removeProductFromBrowsedProducts', product)
    context.dispatch('getBrontoRecommendedProducts')
    context.dispatch('getStuartRecommendedProducts')
  },
  browse (context, product) {
    // 1.  prepend currentProduct to browsedProducts
    // 2.  delete currentProduct ?
    // 2.  set currentProduct to the input paramter: productId
    // 3.  fire off getBrontoRecommendedProducts
    // 4.  fire off getStuartRecommendedProducts
    //
    // note ^ components should be listening for changes to:
    //     currentProduct, brontoRecommendedProducts, and stuartRecommendedProducts, browsedProducts, and queriedProducts

    console.log(product)
    context.commit('prependProductToBrowsedProducts', context.state.currentProduct)
    context.commit('setCurrentProduct', product)
    context.dispatch('getBrontoRecommendedProducts')
    context.dispatch('getStuartRecommendedProducts')
    // context.commit('setIsLoadingProductsSearch', true)
    // context.commit('setQueriedProducts', {})
    // return new Promise((resolve, reject) => {
    //   axios.get(`${server}/pythonapi/search_catalog?n=15&q=${query}`) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`
    //     .then(function (res) {
    //       context.commit('setIsLoadingProductsSearch', false)
    //       dodisplay(res)
    //       context.commit('setQueriedProducts', res.data)
    //     })
    //     .catch(function (err) {
    //       console.log(err)
    //     })
    // })
  },
  //
// @app.route('/pythonapi/recommendations/ff')
// productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")
// numProductsToRecommend = request.args.get('n', default = 1, type = int)
// doReturnDetails = request.args.get('return_details', default = "true", type = str)
//
// @app.route('/pythonapi/recommendations/bronto')
// productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")
// numProductsToRecommend = request.args.get('n', default = 1, type = int)
// doReturnDetails = request.args.get('return_details', default = "true", type = str)
  //
  // TODO get these from browsed products list.  where/how to execute? fron context.state i believe
  getBrontoRecommendedProducts ({commit, state}) {
    commit('setBrontoRecommendedProducts', [])

    console.log('state.currentProduct')
    console.log(state.currentProduct)
    console.log(state.currentProduct.productId)
    console.log(state.currentProduct['productId'])

    var inputProducts

    if (state.currentProduct.productId === undefined) {
      if (state.browsedProducts.length === 0) {
        return
      } else {
        inputProducts = state.browsedProducts
      }
    } else {
      inputProducts = [state.currentProduct, ...state.browsedProducts]
    }
    commit('setIsLoadingBrontoRecommendedProducts', true)

    console.log('inputProducts')
    console.log(inputProducts)
    // const inputProductIds = inputProducts.map(a => a.productId)
    // console.log('inputProductIds')
    // console.log(inputProductIds)

    var productsStr = ''

    console.log('looping')
    console.log('inputProducts')
    console.log(inputProducts)
    // for (var product in inputProducts) {
    //   console.log('product')
    //   console.log(product)
    //   if (product['productId'] !== undefined && product['productId'].size() > 0) {
    //     productsStr += product['productId']
    //   }
    // }

    for (var i = 0; i < inputProducts.length; i++) {
      var product = inputProducts[i]
      console.log(product)
      if (product['productId'] !== undefined && product['productId'].length > 0) {
        productsStr += product['productId'] + '|'
      }
    }
    // TODO fts - USE JSON - NOT URI fts
    console.log('productsStr')
    console.log(productsStr)
    const url = `${server}/pythonapi/recommendations/bronto?n=5&productIds=${productsStr}`
    console.log('getBrontoRecommendedProducts url: ')
    console.log(url)
    return new Promise((resolve, reject) => {
      axios.get(url) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`
        .then(function (res) {
          commit('setIsLoadingBrontoRecommendedProducts', false)
          console.log('getBrontoRecommendedProducts')
          // dodisplay(res)
          commit('setBrontoRecommendedProducts', res.data)
        })
        .catch(function (err) {
          console.log(err)
        })
    })
  },
  getStuartRecommendedProducts ({commit, state}) {
    commit('setStuartRecommendedProducts', [])

    console.log('state.currentProduct')
    console.log(state.currentProduct)
    console.log(state.currentProduct.productId)
    console.log(state.currentProduct['productId'])

    var inputProducts

    if (state.currentProduct.productId === undefined) {
      if (state.browsedProducts.length === 0) {
        return
      } else {
        inputProducts = state.browsedProducts
      }
    } else {
      inputProducts = [state.currentProduct, ...state.browsedProducts]
    }
    commit('setIsLoadingStuartRecommendedProducts', true)

    var productsStr = ''

    console.log('looping')
    console.log('inputProducts')
    console.log(inputProducts)
    // for (var product in inputProducts) {
    //   console.log('product')
    //   console.log(product)
    //   if (product['productId'] !== undefined && product['productId'].size() > 0) {
    //     productsStr += product['productId']
    //   }
    // }

    for (var i = 0; i < inputProducts.length; i++) {
      var product = inputProducts[i]
      console.log(product)
      if (product['productId'] !== undefined && product['productId'].length > 0) {
        productsStr += product['productId'] + '|'
      }
    }
    // TODO fts - USE JSON - NOT URI fts
    console.log('productsStr')
    console.log(productsStr)
    const url = `${server}/pythonapi/recommendations/ff?n=5&productIds=${productsStr}`
    console.log('getBrontoRecommendedProducts url: ')
    console.log(url)
    return new Promise((resolve, reject) => {
      axios.get(url) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`
        .then(function (res) {
          commit('setIsLoadingStuartRecommendedProducts', false)
          console.log('getStuartRecommendedProducts')
          // dodisplay(res)
          commit('setStuartRecommendedProducts', res.data)
        })
        .catch(function (err) {
          console.log(err)
        })
    })
  }
}

export default new Vuex.Store({
  state,
  getters,
  mutations,
  actions
})

