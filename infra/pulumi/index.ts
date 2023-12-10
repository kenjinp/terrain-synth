import * as aws from "@pulumi/aws"
import * as pulumi from "@pulumi/pulumi"
import * as fs from "fs"
import * as mime from "mime"
import * as path from "path"

// Import the program's configuration settings.
const config = new pulumi.Config()
const includeWWW = config.get("includeWWW")
const domainName = config.get("targetDomain")!
const zoneName = config.get("zoneName")!
const contentPath = config.get("path")!
const indexDocument = config.get("indexDocument") || "index.html"
const errorDocument = config.get("errorDocument") || "error.html"
const stackName = pulumi.getStack()

console.log({
  includeWWW,
  domainName,
  zoneName,
  contentPath,
  indexDocument,
  errorDocument,
  stackName,
})

const bucketName = `${domainName}-${stackName}`

// Create an S3 bucket and configure it as a website.
const bucket = new aws.s3.BucketV2(domainName, {
  bucket: bucketName,
})

// Make sure its publicly accessible
new aws.s3.BucketPublicAccessBlock(`${bucketName}-PublicAccessBlock`, {
  bucket: bucket.id,
  blockPublicAcls: false,
  blockPublicPolicy: false,
  ignorePublicAcls: false,
  restrictPublicBuckets: false,
})

// Configure the bucket as a 'static website'
const bucketWebsiteconfiguration = new aws.s3.BucketWebsiteConfigurationV2(
  `${bucketName}-website-config`,
  {
    bucket: bucket.id,
    indexDocument: {
      suffix: indexDocument,
    },
    errorDocument: {
      key: errorDocument,
    },
  },
)

// Create an S3 Bucket Policy to allow public read of all objects in bucket
// This reusable function can be pulled out into its own module
function publicReadPolicyForBucket(bucketName: string) {
  return JSON.stringify({
    Version: "2012-10-17",
    Statement: [
      {
        Effect: "Allow",
        Principal: "*",
        Action: ["s3:GetObject"],
        Resource: [
          `arn:aws:s3:::${bucketName}/*`, // policy refers to bucket name explicitly
        ],
      },
    ],
  })
}

// Set the access policy for the bucket so all objects are readable
new aws.s3.BucketPolicy("bucketPolicy", {
  bucket: bucket.bucket, // depends on siteBucket -- see explanation below
  policy: bucket.bucket.apply(publicReadPolicyForBucket),
  // transform the siteBucket.bucket output property -- see explanation below
})

// recursively add all our website contents
const addFolderContents = (siteDir: string, prefix?: string) => {
  for (const item of fs.readdirSync(siteDir)) {
    const filePath = path.join(siteDir, item)
    const isDir = fs.lstatSync(filePath).isDirectory()

    // This handles adding subfolders and their content
    if (isDir) {
      const newPrefix = prefix ? path.join(prefix, item) : item
      addFolderContents(filePath, newPrefix)
      continue
    }

    let itemPath = prefix ? path.join(prefix, item) : item
    itemPath = itemPath.replace(/\\/g, "/") // convert Windows paths to something S3 will recognize
    pulumi.log.info(`found and shipping asset ${filePath}`)

    new aws.s3.BucketObjectv2(itemPath, {
      bucket: bucket.id,
      source: new pulumi.asset.FileAsset(filePath), // use FileAsset to point to a file
      contentType: mime.getType(filePath) || undefined, // set the MIME type of the file
    })
  }
}

addFolderContents(contentPath) // base directory for content files

// BEGIN provision ACM certificate
// this will be used to verify we own the domain and can secure it with SSL
const tenMinutes = 60 * 10

const eastRegion = new aws.Provider("east", {
  profile: aws.config.profile,
  region: "us-east-1", // Per AWS, ACM certificate must be in the us-east-1 region.
})

// if includeWWW include required subjectAlternativeNames to support the www subdomain
const certificateConfig: aws.acm.CertificateArgs = {
  domainName,
  validationMethod: "DNS",
  subjectAlternativeNames: includeWWW ? [`www.${domainName}`] : [],
}

const certificate = new aws.acm.Certificate("certificate", certificateConfig, {
  provider: eastRegion,
})

const domainParts = {
  subdomain: "",
  parentDomain: zoneName,
}
const hostedZoneId = aws.route53
  .getZone({ name: domainParts.parentDomain }, { async: true })
  .then(zone => zone.zoneId)

/**
 *  Create a DNS record to prove that we _own_ the domain we're requesting a certificate for.
 *  See https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-validate-dns.html for more info.
 */
const certificateValidationDomain = new aws.route53.Record(
  `${domainName}-validation`,
  {
    name: certificate.domainValidationOptions[0].resourceRecordName,
    zoneId: hostedZoneId,
    type: certificate.domainValidationOptions[0].resourceRecordType,
    records: [certificate.domainValidationOptions[0].resourceRecordValue],
    ttl: tenMinutes,
  },
)

// if includeWWW ensure we validate the www subdomain as well
let subdomainCertificateValidationDomain
if (includeWWW) {
  subdomainCertificateValidationDomain = new aws.route53.Record(
    `${domainName}-validation2`,
    {
      name: certificate.domainValidationOptions[1].resourceRecordName,
      zoneId: hostedZoneId,
      type: certificate.domainValidationOptions[1].resourceRecordType,
      records: [certificate.domainValidationOptions[1].resourceRecordValue],
      ttl: tenMinutes,
    },
  )
}

// if includeWWW include the validation record for the www subdomain
const validationRecordFqdns =
  subdomainCertificateValidationDomain === undefined
    ? [certificateValidationDomain.fqdn]
    : [
        certificateValidationDomain.fqdn,
        subdomainCertificateValidationDomain.fqdn,
      ]

/**
 * This is a _special_ resource that waits for ACM to complete validation via the DNS record
 * checking for a status of "ISSUED" on the certificate itself. No actual resources are
 * created (or updated or deleted).
 *
 * See https://www.terraform.io/docs/providers/aws/r/acm_certificate_validation.html for slightly more detail
 * and https://github.com/terraform-providers/terraform-provider-aws/blob/master/aws/resource_aws_acm_certificate_validation.go
 * for the actual implementation.
 */
const certificateValidation = new aws.acm.CertificateValidation(
  "certificateValidation",
  {
    certificateArn: certificate.arn,
    validationRecordFqdns: validationRecordFqdns,
  },
  { provider: eastRegion },
)

const headersPolicy = new aws.cloudfront.ResponseHeadersPolicy("cdnHeaders", {
  customHeadersConfig: {
    items: [
      {
        header: "Cross-Origin-Opener-Policy",
        override: true,
        value: "same-origin",
      },
      {
        header: "Cross-Origin-Embedder-Policy",
        override: true,
        value: "require-corp",
      },
    ],
  },
})

// Create a CloudFront CDN to distribute and cache the website.
const cdn = new aws.cloudfront.Distribution("cdn", {
  enabled: true,
  isIpv6Enabled: true,
  aliases: [domainName],
  origins: [
    {
      originId: bucket.arn,
      domainName: bucketWebsiteconfiguration.websiteEndpoint,

      customOriginConfig: {
        originProtocolPolicy: "http-only",
        httpPort: 80,
        httpsPort: 443,
        originSslProtocols: ["TLSv1.2"],
      },
    },
  ],
  // defaultRootObject: `/${indexDocument}`, experimentally, this isn't required
  defaultCacheBehavior: {
    responseHeadersPolicyId: headersPolicy.id,
    targetOriginId: bucket.arn,
    viewerProtocolPolicy: "redirect-to-https",
    allowedMethods: ["GET", "HEAD", "OPTIONS"],
    cachedMethods: ["GET", "HEAD", "OPTIONS"],
    defaultTtl: 600,
    maxTtl: 600,
    minTtl: 600,
    forwardedValues: {
      queryString: true,
      cookies: {
        forward: "all",
      },
    },
  },
  priceClass: "PriceClass_100",
  customErrorResponses: [
    {
      errorCode: 404,
      responseCode: 404,
      responsePagePath: `/${errorDocument}`,
    },
  ],
  restrictions: {
    geoRestriction: {
      restrictionType: "none",
    },
  },
  viewerCertificate: {
    sslSupportMethod: "sni-only",
    acmCertificateArn: certificateValidation.certificateArn,
    cloudfrontDefaultCertificate: true,
  },
})

// Finally create an A record for our domain that directs to our custom domain.
new aws.route53.Record("webDnsRecord", {
  name: domainName,
  type: "A",
  zoneId: hostedZoneId,
  aliases: [
    {
      evaluateTargetHealth: true,
      name: cdn.domainName,
      zoneId: cdn.hostedZoneId,
    },
  ],
})

// Export the URLs and hostnames of the bucket and distribution.
export const originURL = pulumi.interpolate`http://${bucket.bucketDomainName}`
export const originHostname = bucket.bucketRegionalDomainName
export const cdnURL = pulumi.interpolate`https://${cdn.domainName}`
export const cdnHostname = cdn.domainName
export const bucketRegionalDomainName = pulumi.interpolate`http://${bucket.bucketRegionalDomainName}`
export const bucketDomainName = pulumi.interpolate`http://${bucket.bucketDomainName}`
export const website = `https://${domainName}`
export const bucketWebsiteConfigurationDomain = pulumi.interpolate`http://${bucketWebsiteconfiguration.websiteDomain}`
export const bucketWebsiteConfigurationEndpoint = pulumi.interpolate`http://${bucketWebsiteconfiguration.websiteEndpoint}`
