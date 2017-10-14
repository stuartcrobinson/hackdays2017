import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'

Vue.use(Vuex)

const state = {
  count: 0,
  name: 'stuart',
  gifurl: '',
  isyes: false,
  isLoadingProductsSearch: false,
  queriedProducts: []
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
  setisLoadingProductsSearch (state, booleanvalue) {
    state.isLoadingProductsSearch = booleanvalue
  },
  setQueriedProducts (state, queriedProducts) {
    state.queriedProducts = queriedProducts
  },
  setanything (state, variable, value) {
    // state[variable] = value
    Vue.set(state, variable, value)
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
  loadproducts (context, query) {
    context.commit('setisLoadingProductsSearch', true)

    return new Promise((resolve, reject) => {
      axios.get(`/api/queryproducts/${query}`) // `http://robinson.brontolabs.local:9115/products/_search?q=${query}`
        .then(function (res) {
          context.commit('setisLoadingProductsSearch', false)
          console.log('res:')
          console.log(res)
          console.log(' res.data.hits.hits:')
          console.log(res.data.hits.hits)
          context.commit('setQueriedProducts', res.data.hits.hits)
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

//   axios.post(Config.baseURL + '/api/customer/' + customer.id + '/update', customer)
//     .then((response) => {
//       context.commit('UPDATE_CUSTOMER', customer)
//       resolve()
//     }).catch((response) => {
//     reject()
//   })
// })
//
// ,
// updatecustomer2 (context, customer) {
//   return axios.post(Config.baseURL + '/api/customer/' + customer.id + '/update', customer)
//     .then((response) => {
//       context.commit('UPDATE_CUSTOMER', customer)
//     })
// }
