import { app } from '../../scripts/app.js'

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r
        };
    } else {
        object[property] = callback;
    }
}
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

app.registerExtension({
    name: "ORPHEUS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name == "ORPH_Combine") {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                chainCallback(this, "onConnectionsChange", function (contype, slot, iscon, linfo) {
                    if (contype == LiteGraph.INPUT) {
                        if (iscon && linfo && slot+1 == this.inputs.length) {
                            this.addInput('prompt_' + (slot+2), "ORPH_TOKENS")
                        } else if (!iscon && slot+2 == this.inputs.length) {
                            this.removeInput(slot+1)
                            fitHeight(this)
                        }
                    }
                })
            })
        }
    },
})
