<template>
  <div :class="component_class">
    <form @submit.prevent="onsubmit">
      <label :for="input_id">{{ label }}</label>
      <input
        type="text"
        v-model="inputvmodel"
        :id="input_id"
        :list="datalist_id"
        :autocomplete="autocomplete"
        v-on:input="inputinputevent"
        :placeholder="placeholder"
        :disabled="disabled"
        :autofocus="autofocus"
      />
      <datalist :id="datalist_id">
        <option v-for="item in typeaheads">{{ item }}</option>
      </datalist>
    </form>
  </div>
</template>

<script>
  import axios from 'axios'
  import _ from 'lodash'

  export default {
    props: {
      label: {
        type: String,
        default: ''
      },
      /* placeholder! */
      placeholder: {
        type: String,
        default: ''
      },
      wait: {
        type: Number,
        default: 100
      },
      get_typeahead_endpoint_full: Function,
      typeahead_endpoint_partial: String,
      onsubmit: {
        type: Function,
        required: true
      },
      input_id: {
        type: String,
        default: 'v_datalist_autocomplete_input_id'
      },
      datalist_id: {
        type: String,
        default: 'v_datalist_autocomplete_datalist_id'
      },
      component_class: {
        type: String,
        default: 'v_datalist_autocomplete_class'
      },
      initial_value: {
        type: String,
        default: ''
      },
      min_length: {
        type: Number,
        default: 1
      },
      autocomplete: {
        type: Boolean,
        default: false
      },
      autofocus: {
        type: Boolean,
        default: false
      },
      disabled: {           //TODO test these input parameters.  add: multiple? pattern? readonly?
        type: Boolean,
        default: false
      }
    },

    methods: {
      inputinputevent () {
        const prevInputValue = this.valuehistory[1]
        const currInputValue = this.inputvmodel

        if (this.typeaheads.includes(currInputValue) && (prevInputValue.length > currInputValue.length + 1 || currInputValue.length > prevInputValue.length + 1)) {
          this.onsubmit(this.inputvmodel)
          this.currentInputValueWasSelectedFromDropdown = true
        } else {
          this.currentInputValueWasSelectedFromDropdown = false
        }
      },
      getTypeaheads: _.debounce(
        function () {
          if (this.inputvmodel.length === 0) {
            this.typeaheads = []
          } else {
            if (this.inputvmodel.length >= this.min_length) {
              var vm = this

              const typeaheadEndpoint = this.typeahead_endpoint_partial.length > 0 ? this.typeahead_endpoint_partial + this.inputvmodel : this.get_typeahead_endpoint(this.inputvmodel)

              axios.get(typeaheadEndpoint)
                .then(function (response) {
                  vm.typeaheads = response.data
                  document.getElementById(vm.datalist_id).focus()
                })
                .catch(function (error) {
                  console.log(error)
                })
            }
          }
        },
        this.wait // milliseconds to wait for the user to stop typing. https://lodash.com/docs#debounce
      )
    },
    watch: {
      inputvmodel: function () {
        if (!this.currentInputValueWasSelectedFromDropdown) {
          this.getTypeaheads()
        }
        this.valuehistory = [this.inputvmodel, ...this.valuehistory].slice(0, 2)
      }
    },
    data () {
      return {
        inputvmodel: this.initial_value,
        typeaheads: [],
        valuehistory: ['wefaiwuehfliawuhef', 'gser98ugefawf'],
        currentInputValueWasSelectedFromDropdown: false
      }
    }
  }
</script>

<style>
  /* Hide the list on focus of the input field */
  datalist {
    display: none;
  }

  /* specifically hide the arrow on focus */
  input::-webkit-calendar-picker-indicator {
    display: none;
  }
</style>
