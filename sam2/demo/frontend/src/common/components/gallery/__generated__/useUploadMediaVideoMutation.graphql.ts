/**
 * @generated SignedSource<<97a24db9fbfc3cfbead139c8bbd3ac23>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type useUploadMediaVideoMutation$variables = {
  file: any;
};
export type useUploadMediaVideoMutation$data = {
  readonly uploadVideo: {
    readonly height: number;
    readonly id: any;
    readonly path: string;
    readonly posterPath: string | null | undefined;
    readonly posterUrl: string;
    readonly url: string;
    readonly width: number;
  };
};
export type useUploadMediaVideoMutation = {
  response: useUploadMediaVideoMutation$data;
  variables: useUploadMediaVideoMutation$variables;
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
    "name": "uploadVideo",
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
    "name": "useUploadMediaVideoMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "useUploadMediaVideoMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "7a60fb499e5f257c8bef44d5f5f64715",
    "id": null,
    "metadata": {},
    "name": "useUploadMediaVideoMutation",
    "operationKind": "mutation",
    "text": "mutation useUploadMediaVideoMutation(\n  $file: Upload!\n) {\n  uploadVideo(file: $file) {\n    id\n    height\n    width\n    url\n    path\n    posterPath\n    posterUrl\n  }\n}\n"
  }
};
})();

(node as any).hash = "52d8915d027fbb534a5dee7b3b515a6c";

export default node;
