<template>
  <div>

    START SEARCHFIELD COMPONENT
    {{ msg }} nickname: {{ nickname }}

    <!--<autocomplete-->
    <!--value="myvalue"-->
    <!--classes="autocompleteinputclass"-->
    <!--placeholder="Search Products"-->
    <!--remote="/api/queryproducttitletypeaheads/"-->
    <!--@selected="addDistributionGroup">-->
    <!--</autocomplete>-->

    <!--<form @submit="queryProductsMethod">-->
    <form @submit.prevent="queryProductsMethod">
      <label for="nameId">Enter name:</label>
      <input type="text" v-model="namee" id="nameId" list="mydatalist" autocomplete="off" v-on:input="inputinputevent" placeholder="search products"/>

      <br> history: {{ valuehistory }} currentInputValueWasSelectedFromDropdown: {{ currentInputValueWasSelectedFromDropdown }}

      <datalist id="mydatalist">
        <option v-for="item in typeaheads">{{item}}</option>
      </datalist>

      <br>
      <!--<div v-for="item in typeaheads">{{item}}</div>-->

      <br>
      <!--typeaheadlistid: {{typeaheadlistid}} <br>-->
      typeaheads: {{typeaheads}} <br>

    </form>

    <p>{{ namee }} is {{ age }} years old.</p>

    END SEARCHFIELD COMPONENT

  </div>
</template>
<!--https://github.com/vuejs/vuejs.org/blob/master/src/v2/guide/computed.md-->

<!--https://paliari.github.io/v-autocomplete/-->

<!--this one sucks cos you can only search for stuff in the dropdown wtf-->
<!--import Autocomplete from 'vuejs-auto-complete'--> <!--https://www.npmjs.com/package/vuejs-auto-complete-->
<!--<autocomplete-->
<!--inputClass="autocompleteinputclass"-->
<!--placeholder="Search Products"-->
<!--source="/api/queryproducttitletypeaheads/"-->
<!--:results-display="formattedDisplay"-->
<!--@selected="addDistributionGroup">-->
<!--</autocomplete>-->

<script>
  import { mapActions } from 'vuex'
  import axios from 'axios'
  //  import Autocomplete from 'vuejs-autocomplete'
  import _ from 'lodash'
  //  import $ from 'jquery'
  //  import VueLodash from 'vue-lodash'

  //  Vue.use(VueLodash, lodash)
  //
  //  // constructs the suggestion engine
  //  var states = new Bloodhound({
  //    datumTokenizer: Bloodhound.tokenizers.whitespace,
  //    queryTokenizer: Bloodhound.tokenizers.whitespace,
  //    // `states` is an array of state names defined in "The Basics"
  //    local: states
  //  });
  //
  //  $('#bloodhound .typeahead').typeahead({
  //      hint: true,
  //      highlight: true,
  //      minLength: 1
  //    },
  //    {
  //      name: 'states',
  //      source: states
  //    });
  //  function getElementByXpath (path) {
  //    return document.evaluate(path, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue
  //  }
  export default {
    props: {
      nickname: {
        default: 'ned',
        type: String
      }
    },
    mounted: function () {
      var el = document.getElementById('nameId')

      el.onkeydown = function (evt) {
        evt = evt || window.event
        console.log('keydown: ' + evt.keyCode)  // String.fromCharCode()
        console.log('keydown: ' + String.fromCharCode(evt.keyCode))  //
      }

      el.onkeyup = function (evt) {
        evt = evt || window.event
        console.log('keyup: ' + evt.keyCode)
        console.log('keyup: ' + String.fromCharCode(evt.keyCode))
      }

      // `this` points to the vm instance
//      document.getElementById('nameId').addEventListener('input', function () {
//        console.log('changed')
//      })
      document.getElementById('nameId').addEventListener('select', function () {
        console.log('select')
      })
//      document.getElementById('nameId').bind('select', function () {
//        console.log('selected')
//      })
      console.log('created')
    },
//    components: {
//      Autocomplete
//    },
    methods: {
      ...mapActions([
        'loadproducts'
      ]),
      inputinputevent () {
        console.log('input event!')
        const prevInputValue = this.valuehistory[1]
        const currInputValue = this.namee

        if (prevInputValue.length > currInputValue.length + 1 || currInputValue.length > prevInputValue.length + 1) {
          if (this.typeaheads.includes(currInputValue)) {
            console.log('contained!!!!!!!!!!!!!!!!')
            this.typeaheads = []
            this.queryProductsMethod()
            this.currentInputValueWasSelectedFromDropdown = true
            return
          }
        }
        this.currentInputValueWasSelectedFromDropdown = false
      },
      doalert () {
        alert('hi')
      },
      queryProductsMethod () {
        console.log('querying products for ' + this.namee)
        console.log('querying products for ' + JSON.stringify(this.namee))
        this.$store.dispatch('loadproducts', this.namee)
        this.typeaheads = []
        console.log('just cleared typeaheads')
        console.log('typeaheads:')
        console.log(this.typeaheads)

//        const input = document.getElementsByClassName('autocompleteinputclass')[0].getElementsByTagName('input')[0]
//
//        input.setAttribute('value', 'stuart')
//
//        input.focus()
//
//        console.log(input)

        // console.log( getElementByXpath("//html[1]/body[1]/div[1]") );

//        getElementByXpath('//div[@class=\'autocomplete__box\']//input[@type=\'text\']').value = 'myvalue'

//        input.value = 'myvalue'
//
//        console.log(input)
      },
//      ,
//      queryTypeaheadMethod () {
//        this.typeaheadlistid = 'active'
//        this.$store.dispatch('loadtypeaheads', this.namee)
//      }
      //
      // _.debounce is a function provided by lodash to limit how
      // often a particularly expensive operation can be run.
      // In this case, we want to limit how often we access
      // yesno.wtf/api, waiting until the user has completely
      // finished typing before making the ajax request. To learn
      // more about the _.debounce function (and its cousin
      // _.throttle), visit: https://lodash.com/docs#debounce
      getTypeaheads: _.debounce(
        function () {
          console.log('in getTypeaheads')
          if (this.namee.length === 0) {
            this.typeaheads = []
          } else {
            var vm = this
            axios.get(`/api/queryproducttitletypeaheads/${vm.namee}`)
              .then(function (response) {
                vm.typeaheads = response.data

                console.log('new typeaheads: ' + vm.typeaheads)

//                const el = document.activeElement

                document.getElementById('mydatalist').focus()

//                el.focus()
              })
              .catch(function (error) {
                console.log(error)
              })
          }
        },
        200 // This is the number of milliseconds we wait for the user to stop typing.
      )
      // from https://www.npmjs.com/package/vuejs-auto-complete

//      distributionGroupsEndpoint () {
//        return `/api/queryproducttitletypeaheads/${this.namee}`
//      },
//      addDistributionGroup (group) {
//        console.log('selected')
//        this.namee = group.selectedObject
//        this.queryProductsMethod()
//        // access the autocomplete component methods from the parent
//        this.$refs.autocomplete.clearValues()
//      },
// //      authHeaders () {
// //        return {
// //          'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1Ni.... unused - going to local express proxy thing'
// //        }
// //      },
//       formattedDisplay (result) {
//        return result.length + ' ' + result
//      }
    },
    watch: {
      // whenever question changes, this function will run
      namee: function (whatisthisfor) {
//        console.log('watch: namee changed')
        if (!this.currentInputValueWasSelectedFromDropdown) {
          this.getTypeaheads()
        }
//        this.typeaheadurl = `/api/queryproducttitletypeaheads/${this.namee}`
        this.valuehistory = [this.namee, ...this.valuehistory]
//        console.log('valuehist')
//        console.log(this.valuehistory)
//
        this.valuehistory = this.valuehistory.slice(0, 2)
//        console.log(this.valuehistory)
      }
    },
    data () {
      return {
        msg: 'Welcome to Your Vue.js App',
        namee: 'Ashley',
        age: 100,
        typeaheads: [], // '1', '2', '3', '4', 'hot dogs', 'star wars', 'plates', 'napkins', 'corn', 'chips'
        typeaheadurl: '',
        valuehistory: ['wefaiwuehfliawuhef', 'gser98ugefawf'],
        currentInputValueWasSelectedFromDropdown: false
      }
    }
  }
</script>

<style>
  /*!* Hide the list on focus of the input field *!*/
  /*datalist {*/
  /*display: none;*/
  /*}*/

  /* specifically hide the arrow on focus */
  input::-webkit-calendar-picker-indicator {
    display: none;
  }
</style>
