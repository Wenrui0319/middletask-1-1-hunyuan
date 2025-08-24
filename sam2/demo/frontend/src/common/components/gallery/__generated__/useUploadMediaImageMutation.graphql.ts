/**
 * @generated SignedSource<<d65874aa497b6d78f42f9630f79e5bbe>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type useUploadMediaImageMutation$variables = {
  file: any;
};
export type useUploadMediaImageMutation$data = {
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
export type useUploadMediaImageMutation = {
  response: useUploadMediaImageMutation$data;
  variables: useUploadMediaImageMutation$variables;
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
    "name": "useUploadMediaImageMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "useUploadMediaImageMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "b3eefb3cd68c4c4abbd5d9496228abda",
    "id": null,
    "metadata": {},
    "name": "useUploadMediaImageMutation",
    "operationKind": "mutation",
    "text": "mutation useUploadMediaImageMutation(\n  $file: Upload!\n) {\n  uploadImage(file: $file) {\n    id\n    height\n    width\n    url\n    path\n    posterPath\n    posterUrl\n  }\n}\n"
  }
};
})();

(node as any).hash = "cd77643ee1ab470e5e7ec128324dcfda";

export default node;
