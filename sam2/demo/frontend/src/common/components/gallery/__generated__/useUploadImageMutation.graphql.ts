/**
 * @generated SignedSource<<139eb691533caf5d285704fd51c50a4e>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type useUploadImageMutation$variables = {
  file: any;
};
export type useUploadImageMutation$data = {
  readonly uploadImage: {
    readonly height: number;
    readonly id: any;
    readonly path: string;
    readonly posterPath: string | null | undefined;
    readonly posterUrl: string;
    readonly url: string;
    readonly width: number;
  };
};
export type useUploadImageMutation = {
  response: useUploadImageMutation$data;
  variables: useUploadImageMutation$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "file"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "file",
        "variableName": "file"
      }
    ],
    "concreteType": "Video",
    "kind": "LinkedField",
    "name": "uploadImage",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "id",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "height",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "width",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "url",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "path",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "posterPath",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "posterUrl",
        "storageKey": null
      }
    ],
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "useUploadImageMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "useUploadImageMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "71f79bbc986285a8f5d297b8c17b8463",
    "id": null,
    "metadata": {},
    "name": "useUploadImageMutation",
    "operationKind": "mutation",
    "text": "mutation useUploadImageMutation(\n  $file: Upload!\n) {\n  uploadImage(file: $file) {\n    id\n    height\n    width\n    url\n    path\n    posterPath\n    posterUrl\n  }\n}\n"
  }
};
})();

(node as any).hash = "24f1aa61d7c0139f779cdb4a8ebf8f4f";

export default node;
